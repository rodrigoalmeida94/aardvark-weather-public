import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from aardvark.architectures import MLP
from aardvark.set_convs import convDeepSet
from aardvark.unet_wrap_padding import *
from aardvark.vit import *

sys.path.append("../")


class ConditionalLayerNorm(nn.Module):
    """LayerNorm whose scale and shift are predicted from a noise embedding.

    Initialised so it acts as identity at the start of training
    (zero-init projection keeps scale=0, shift=0 → output = LN(x)).
    """

    def __init__(self, dim: int, noise_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(noise_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        modulation = self.proj(noise_emb)
        scale, shift = modulation.chunk(2, dim=-1)
        return (1 + scale) * x_norm + shift


class NoiseBlock(nn.Module):
    """Wraps a timm ViT Block, replacing its LayerNorms with ConditionalLayerNorm."""

    def __init__(self, block: nn.Module, noise_dim: int):
        super().__init__()
        dim = block.norm1.normalized_shape[0]
        self.attn = block.attn
        self.mlp = block.mlp
        self.ls1 = block.ls1
        self.ls2 = block.ls2
        self.drop_path1 = block.drop_path1
        self.drop_path2 = block.drop_path2
        self.norm1 = ConditionalLayerNorm(dim, noise_dim)
        self.norm2 = ConditionalLayerNorm(dim, noise_dim)

    def forward(self, x: torch.Tensor, noise_emb: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x, noise_emb))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x, noise_emb))))
        return x


class ViTCheckpointed(ViT):
    """ViT with optional gradient checkpointing and noise conditioning.

    Args:
        use_noise_conditioning: When True, per-patch Gaussian noise is sampled
            each forward pass and injected according to noise_mode.
        noise_channels: Dimension of the raw noise vector per patch.
        noise_mode: "norm" — noise conditions the pre-norm LayerNorms via
            ConditionalLayerNorm. "embedding" — noise is added directly to
            patch embeddings before the transformer blocks.
    """

    def __init__(
        self,
        *args,
        use_noise_conditioning: bool = False,
        noise_channels: int = 16,
        noise_mode: str = "norm",
        noise_terrain_cond: bool = False,
        n_terrain: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_checkpointing = False
        self.use_noise_conditioning = use_noise_conditioning
        self.noise_mode = noise_mode
        self.noise_terrain_cond = noise_terrain_cond

        if use_noise_conditioning:
            embed_dim = self.blocks[0].attn.qkv.in_features
            self.noise_channels = noise_channels
            self.noise_mlp = nn.Sequential(
                nn.Linear(noise_channels, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            if noise_mode == "norm":
                self.blocks = nn.ModuleList(
                    [NoiseBlock(blk, noise_dim=embed_dim) for blk in self.blocks]
                )
            if noise_terrain_cond:
                # Terrain-gated noise amplitude: a per-patch positive scale g(terrain)
                # multiplies noise_emb, so the injected spread is heteroscedastic —
                # larger over rough/land patches (high sdor), smaller over smooth
                # ocean. Init to g ≡ 1 (zero-weight final layer, softplus-bias = 1)
                # so a warm-started run is unchanged at step 0 and *learns* the
                # terrain dependence under CRPS.
                self.noise_scale_mlp = nn.Sequential(
                    nn.Linear(n_terrain, 16),
                    nn.SiLU(),
                    nn.Linear(16, 1),
                    nn.Softplus(),
                )
                nn.init.zeros_(self.noise_scale_mlp[2].weight)
                # softplus(b) = 1  ->  b = log(e - 1)
                inv_softplus_one = float(torch.log(torch.expm1(torch.tensor(1.0))))
                nn.init.constant_(self.noise_scale_mlp[2].bias, inv_softplus_one)

    def set_grad_checkpointing(self, enabled=True):
        self.grad_checkpointing = enabled

    def forward_encoder(self, x, lead_times, variables, noise_emb=None):
        if isinstance(variables, list):
            variables = tuple(variables)

        if self.per_var_embedding:
            embeds = []
            var_ids = self.get_var_ids(variables, x.device)
            for i in range(len(var_ids)):
                idx = var_ids[i]
                embeds.append(self.token_embeds[idx](x[:, i : i + 1]))
            x = torch.stack(embeds, dim=1)
            var_embed = self.get_var_emb(self.var_embed, variables)
            x = x + var_embed.unsqueeze(2)
            x = self.aggregate_variables(x)
        else:
            x = self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = self.token_embeds[0](x)

        x = x + self.pos_embed
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        x = x + lead_time_emb.unsqueeze(1)
        x = self.pos_drop(x)

        if noise_emb is not None and self.noise_mode == "embedding":
            x = x + noise_emb
            noise_emb = None

        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                args = (x, noise_emb) if noise_emb is not None else (x,)
                x = checkpoint.checkpoint(blk, *args)
            else:
                x = blk(x, noise_emb) if noise_emb is not None else blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, lead_times=None, film_index=None, terrain=None):
        if lead_times is None:
            lead_times = torch.ones(x.shape[0], device=x.device).float().unsqueeze(-1)

        noise_emb = None
        if self.use_noise_conditioning:
            xi = torch.randn(
                x.shape[0], self.num_patches, self.noise_channels, device=x.device
            )
            noise_emb = self.noise_mlp(xi)
            if self.noise_terrain_cond and terrain is not None:
                # g(terrain): (B, P, 1) positive per-patch amplitude, broadcast over
                # the embed dim — terrain-aware (heteroscedastic) spread.
                #
                # Mean-normalise g across patches so mean_patch(g) == 1: the gate
                # can only *redistribute* spread (more over some patches, less over
                # others), never rescale the global amplitude (that is the job of
                # noise_mlp). Without this, a globally under/over-dispersed ensemble
                # makes CRPS collapse the gate to a constant g != 1, expressing a
                # global volume change instead of the intended terrain structure
                # (observed empirically: the gate drifted to a uniform g < 1). The
                # init g ≡ 1 is unchanged by the normalisation (1 / mean(1) = 1), so
                # a warm-started run is still a no-op at step 0.
                g = self.noise_scale_mlp(terrain)
                g = g / g.mean(dim=1, keepdim=True).clamp_min(1e-6)
                noise_emb = noise_emb * g

        out = self.forward_encoder(x, lead_times[:, 0], self.default_vars, noise_emb=noise_emb)
        preds = self.head(out)
        preds = self.unpatchify(preds)
        return preds.permute(0, 2, 3, 1)


class ConvCNPWeather(nn.Module):
    """
    ConvCNP class used for the encoder and processor modules
    """

    # Observation streams, in the channel order they are concatenated into the
    # encoder input. Per-stream noise (use_stream_noise) is keyed by these names.
    # Deterministic context (elev, climatology, aux_time) is intentionally excluded.
    STREAM_NAMES = (
        "iasi",
        "ascat",
        "hadisd",
        "icoads",
        "sat",
        "amsua",
        "amsub",
        "igra",
        "hirs",
    )

    def __init__(
        self,
        in_channels,
        out_channels,
        int_channels,
        device,
        res,
        data_path="../data/",
        gnp=False,
        mode="assimilation",
        decoder=None,
        film=False,
        two_frames=False,
        use_noise_conditioning: bool = False,
        noise_mode: str = "norm",
        noise_terrain_cond: bool = False,
        use_stream_noise: bool = False,
        stream_noise_init: float = 0.01,
    ):

        super().__init__()

        self.device = device
        # Terrain-gated noise amplitude (only meaningful with noise conditioning).
        self.noise_terrain_cond = bool(noise_terrain_cond) and bool(use_noise_conditioning)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.int_channels = int_channels
        self.decoder = decoder
        self.int_x = 256
        self.int_y = 128
        self.data_path = data_path
        self.mode = mode
        self.film = film
        self.two_frames = two_frames

        # Per-stream aleatoric noise: one learned Gaussian std per observation
        # stream, injected into that stream's gridded encoding before the streams
        # are concatenated. Because each stream occupies its own contiguous channel
        # slice at that point, the perturbation is attributable to a single data
        # source (unlike embed-dim noise, which mixes all streams into patch tokens).
        self.use_stream_noise = use_stream_noise
        if use_stream_noise:
            # Store the std in inverse-softplus space so softplus(param) is the
            # actual std and stays strictly positive; init so softplus(.) ≈ init.
            init = torch.log(torch.expm1(torch.tensor(float(stream_noise_init))))
            self.stream_noise_log_std = nn.ParameterDict(
                {name: nn.Parameter(init.clone()) for name in self.STREAM_NAMES}
            )

        N_SAT_VARS = 2
        N_ICOADS_VARS = 5
        N_HADISD_VARS = 5

        # Load internal grid longitude-latitude locations
        self.era5_x = (
            torch.from_numpy(
                np.load(self.data_path + "grid_lon_lat/era5_x_{}.npy".format(res))
            ).float()
            / 360
        )
        self.era5_y = (
            torch.from_numpy(
                np.load(self.data_path + "grid_lon_lat/era5_y_{}.npy".format(res))
            ).float()
            / 360
        )

        self.int_grid = [
            (torch.linspace(0, 360, 240) / 360).float().cuda(),
            (torch.linspace(-90, 90, 121) / 360).float().cuda(),
        ]

        self.int_grid = [self.int_grid[0].unsqueeze(0), self.int_grid[1].unsqueeze(0)]

        # Create input setconvs for each data modality
        self.ascat_setconvs = convDeepSet(
            0.001, "OnToOn", density_channel=True, device=self.device
        )
        self.amsua_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(13)
        ]
        self.amsub_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(12)
        ]
        self.hirs_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(26)
        ]

        self.sat_setconvs = [
            convDeepSet(0.001, "OnToOn", density_channel=True, device=self.device)
            for _ in range(N_SAT_VARS)
        ]
        self.hadisd_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_HADISD_VARS)
        ]
        self.icoads_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(N_ICOADS_VARS)
        ]
        self.igra_setconvs = [
            convDeepSet(0.001, "OffToOn", density_channel=True, device=self.device)
            for _ in range(24)
        ]

        self.sc_out = convDeepSet(
            0.001, "OnToOff", density_channel=False, device=self.device
        )

        # Instantiate the decoder. Here decoder refers to decoder in a convCNP (i.e the ViT backbone)
        if self.decoder == "vit":
            self.decoder_lr = ViTCheckpointed(
                in_channels=in_channels,
                out_channels=out_channels,
                h_channels=512,
                depth=16,
                patch_size=5,
                per_var_embedding=True,
                img_size=[240, 121],
                use_noise_conditioning=use_noise_conditioning,
                noise_mode=noise_mode,
                noise_terrain_cond=self.noise_terrain_cond,
            )

        elif self.decoder == "vit_assimilation":
            self.decoder_lr = ViTCheckpointed(
                in_channels=256,
                out_channels=out_channels,
                h_channels=512,
                depth=8,
                patch_size=3,
                per_var_embedding=False,
                img_size=[256, 128],
                use_noise_conditioning=use_noise_conditioning,
                noise_mode=noise_mode,
                noise_terrain_cond=self.noise_terrain_cond,
            )

        self.mlp = MLP(
            in_channels=out_channels,
            out_channels=out_channels,
            h_channels=128,
            h_layers=4,
        )
        self.break_next = False

    def encoder_hadisd(self, task, prefix):
        """
        Data preprocessing for HadISD
        """

        encodings = []
        for channel in range(4):
            encodings.append(
                self.hadisd_setconvs[channel](
                    x_in=[
                        task["x_context_hadisd_{}".format(prefix)][channel][:, 0, :],
                        task["x_context_hadisd_{}".format(prefix)][channel][:, 1, :],
                    ],
                    wt=task["y_context_hadisd_{}".format(prefix)][channel].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_sat(self, task, prefix):
        """
        Data preprocessing for Gridsat
        """

        encodings = []
        for channel in range(task["sat_{}".format(prefix)].shape[1]):
            encodings.append(
                self.sat_setconvs[channel](
                    x_in=task["sat_x_{}".format(prefix)],
                    wt=task["sat_{}".format(prefix)][:, channel : channel + 1, ...],
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_icoads(self, task, prefix):
        """
        Data preprocessing for ICOADS
        """

        encodings = []
        for channel in range(5):
            encodings.append(
                self.icoads_setconvs[channel](
                    x_in=task["icoads_x_{}".format(prefix)],
                    wt=task["icoads_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)

        return encodings

    def encoder_amsua(self, task, prefix):
        """
        Data preprocessing for AMSU-A
        """

        encodings = []
        task["amsua_{}".format(prefix)][..., -1] = np.nan
        task["amsua_{}".format(prefix)][task["amsua_{}".format(prefix)] == 0] = np.nan
        for i in range(13):
            encodings.append(
                self.amsua_setconvs[i](
                    x_in=task["amsua_x_{}".format(prefix)],
                    wt=task["amsua_{}".format(prefix)].permute(0, 3, 2, 1)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_amsub(self, task, prefix):
        """
        Data preprocessing for AMSU-B
        """

        encodings = []
        task["amsub_{}".format(prefix)][task["amsub_{}".format(prefix)] == 0] = np.nan
        for i in range(12):
            encodings.append(
                self.amsua_setconvs[i](
                    x_in=task["amsub_x_{}".format(prefix)],
                    wt=task["amsub_{}".format(prefix)].permute(0, 3, 1, 2)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_hirs(self, task, prefix):
        """
        Data preprocessing for HIRS
        """

        encodings = []

        task["hirs_{}".format(prefix)][task["hirs_{}".format(prefix)] == 0] = np.nan
        for i in range(26):
            encodings.append(
                self.hirs_setconvs[i](
                    x_in=task["hirs_x_{}".format(prefix)],
                    wt=task["hirs_{}".format(prefix)].permute(0, 3, 1, 2)[
                        :, i : i + 1, ...
                    ],
                    x_out=self.int_grid,
                )
            )

        encodings = torch.cat(encodings, dim=1)
        return encodings

    def encoder_igra(self, task, prefix):
        """
        Data preprocessing for IGRA
        """

        encodings = []
        for channel in range(24):
            encodings.append(
                self.igra_setconvs[channel](
                    x_in=task["igra_x_{}".format(prefix)],
                    wt=task["igra_{}".format(prefix)][:, channel, :].unsqueeze(1),
                    x_out=self.int_grid,
                )
            )
        encodings = torch.cat(encodings, dim=1)

        return encodings

    def encoder_ascat(self, task, prefix):
        """
        Data preprocessing for ASCAT
        """

        task["ascat_{}".format(prefix)][
            torch.isnan(task["ascat_{}".format(prefix)])
        ] = 0
        e = nn.functional.interpolate(
            task["ascat_{}".format(prefix)].permute(0, 3, 1, 2), size=(240, 121)
        )
        e = torch.flip(e, dims=[-1])
        return e

    def encoder_iasi(self, task, prefix):
        """
        Data preprocessing for IASI
        """

        task["iasi_{}".format(prefix)][torch.isnan(task["iasi_{}".format(prefix)])] = 0
        e = nn.functional.interpolate(
            task["iasi_{}".format(prefix)].permute(0, 3, 1, 2), size=(240, 121)
        )
        e = torch.flip(e, dims=[-1])
        return e

    def stream_noise_stds(self):
        """Return the current learned std per observation stream as a plain dict.

        Useful for attribution: reads off how much noise the model has learned to
        attach to each data source. Returns an empty dict when stream noise is off.
        """
        if not self.use_stream_noise:
            return {}
        return {
            name: nn.functional.softplus(self.stream_noise_log_std[name]).item()
            for name in self.STREAM_NAMES
        }

    def _apply_stream_noise(self, enc, name, override):
        """Add per-stream Gaussian noise to one stream's gridded encoding.

        The std is the learned ``softplus(stream_noise_log_std[name])`` unless
        ``override`` supplies an absolute std for this stream. ``override`` lets an
        attribution sweep silence every stream but one (or scan a stream's std)
        without touching the learned parameters. A std of 0 is a no-op.
        """
        if not self.use_stream_noise:
            return enc
        if override is not None and name in override:
            std = override[name]
            if std == 0:
                return enc
            std = torch.as_tensor(std, device=enc.device, dtype=enc.dtype)
        else:
            std = nn.functional.softplus(self.stream_noise_log_std[name])
        return enc + torch.randn_like(enc) * std

    def _obs_encodings(self, task, prefix, override):
        """Build the observation-stream encodings for one timestep, noised per stream.

        Returns the list in canonical ``STREAM_NAMES`` channel order so the
        concatenated layout (and thus checkpoint compatibility) is unchanged.
        """
        builders = {
            "iasi": self.encoder_iasi,
            "ascat": self.encoder_ascat,
            "hadisd": self.encoder_hadisd,
            "icoads": self.encoder_icoads,
            "sat": self.encoder_sat,
            "amsua": self.encoder_amsua,
            "amsub": self.encoder_amsub,
            "igra": self.encoder_igra,
            "hirs": self.encoder_hirs,
        }
        return [
            self._apply_stream_noise(builders[name](task, prefix), name, override)
            for name in self.STREAM_NAMES
        ]

    def forward(self, task, film_index, noise_std=0.0, stream_noise_override=None):

        # Per-patch terrain feature for the terrain-gated noise amplitude; stays
        # None unless noise_terrain_cond is on (assimilation + vit_assimilation).
        terrain = None

        # Setup input
        if self.mode == "assimilation":

            self.int_grid = [i.to(task["y_target"].device) for i in self.int_grid]
            elev = nn.functional.interpolate(
                torch.flip(task["era5_elev_current"].permute(0, 1, 3, 2), dims=[2]),
                size=(self.int_grid[0].shape[1], self.int_grid[1].shape[1]),
            )
            elev = torch.flip(task["era5_elev_current"].permute(0, 1, 3, 2), dims=[2])

            if self.noise_terrain_cond:
                # elev is (B, C, 240, 121) in the same frame as x; take sdor (ch4,
                # std of sub-grid orography = the ruggedness the gate should key on)
                # and elevation (ch0), match x's interpolation to (256, 128), then
                # average-pool to the ViT patch grid -> (B, num_patches, 2).
                # NB: earlier code used ch2, which is `anor` (anisotropy), not sdor
                # (see scripts/convert_orography.py for the channel layout); ch4 is
                # the intended roughness feature. Pre-existing terrain checkpoints
                # were trained on ch2 and must be retrained after this fix.
                terr = elev[:, [4, 0], :, :]
                terr = nn.functional.interpolate(terr, size=(256, 128))
                gh, gw = self.decoder_lr.token_embeds[0].grid_size
                terr = nn.functional.adaptive_avg_pool2d(terr, (gh, gw))
                terrain = terr.flatten(2).transpose(1, 2)

            context = [
                elev,
                task["climatology_current"],
                torch.ones_like(elev[:, :5, ...])
                * task["aux_time_current"].unsqueeze(-1).unsqueeze(-1),
            ]
            if not self.two_frames:
                encodings = [
                    *self._obs_encodings(task, "current", stream_noise_override),
                    *context,
                ]
            else:
                # Option to pass two timesteps (t=-1 and t=0) as input. The same
                # learned per-stream std is shared across the current/prev frame.
                encodings = [
                    *self._obs_encodings(task, "current", stream_noise_override),
                    *self._obs_encodings(task, "prev", stream_noise_override),
                    *context,
                ]
            x = torch.cat(encodings, dim=1)

        else:
            x = task["y_context"]

        if x.shape[-1] > x.shape[-2]:
            x = x.permute(0, 1, 3, 2)

        if noise_std > 0.0:
            x = x + torch.randn_like(x) * noise_std

        # Run ViT backbone
        if self.decoder == "vit":
            x = self.decoder_lr(x, lead_times=task["lt"])
            x = x.permute(0, 3, 1, 2)
        else:
            x = nn.functional.interpolate(x, size=(256, 128))
            x = self.decoder_lr(x, film_index=(task["lt"] * 0) + 1, terrain=terrain)

        # Process outputs

        if np.logical_and(
            self.mode == "assimilation", self.decoder == "vit_assimilation"
        ):
            x = nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(240, 121))
            return x.permute(0, 3, 2, 1)

        elif self.mode == "forecast":
            x = nn.functional.interpolate(x, size=(240, 121)).permute(0, 2, 3, 1)
            return x.permute(0, 2, 1, 3)

        return x
