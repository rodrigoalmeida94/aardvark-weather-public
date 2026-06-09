import numpy as np
import torch
import torch.nn as nn


class RmseLoss(nn.Module):
    """
    RMSE loss
    """

    def __init__(self, start_ind=0, end_ind=24):

        super().__init__()
        self.start_ind = start_ind
        self.end_ind = end_ind

    def forward(
        self,
        target,
        output,
        prev_step_output,
        fix_sigma=False,
        unwrap=False,
        expand=False,
    ):

        squared_diff = ((target.to(output.device) - output) ** 2)[
            ..., self.start_ind : self.end_ind
        ]
        return torch.mean(torch.sqrt(torch.nanmean(squared_diff, dim=(1, 2, 3))))


class PressureWeightedRmseLoss(nn.Module):
    """
    Latitude weighted pressure weighted RMSE loss used in training the processor
    """

    def __init__(
        self,
        res,
        era5_mode,
        data_dir,
        aux_data_dir,
        weight_per_variable=False,
    ):
        super().__init__()

        self.weights = torch.from_numpy(
            np.load(aux_data_dir + "lat_weights/weights_lat_{}.npy".format(res))[
                np.newaxis, ..., np.newaxis
            ]
        ).float()

        self.weight_per_variable = weight_per_variable
        self.variable_weights = torch.from_numpy(
            np.load(aux_data_dir + "loss_weights.npy")[
                np.newaxis, np.newaxis, np.newaxis, :
            ]
        ).float()

        pressure_file = (
            "era5/era5_pressure_levels_{}_24ch.npy".format(era5_mode)
            if era5_mode == "4u"
            else "era5/era5_pressure_levels_{}.npy".format(era5_mode)
        )
        self.pressure_levels = (
            torch.from_numpy(
                np.load(data_dir + pressure_file)[np.newaxis, np.newaxis, np.newaxis, :]
            ).float()
            / 1000
        )

    def forward(
        self,
        target,
        output,
        prev_step_output,
        fix_sigma=False,
        unwrap=False,
        expand=False,
    ):

        squared_diff = (target.to(output.device) - output) ** 2

        if not expand:
            weighted_sqared_diff = (
                squared_diff
                * self.weights.to(target.device)
                * self.pressure_levels.to(target.device)
            )
            return torch.mean(
                torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2, 3)))
            )

        weighted_sqared_diff = squared_diff * self.weights.to(target.device)

        if self.weight_per_variable:
            weighted_sqared_diff = weighted_sqared_diff * self.variable_weights.to(
                weighted_sqared_diff.device
            )

        return torch.mean(
            torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2))), dim=0
        )


class WeightedRmseLoss(nn.Module):
    """
    Latitude weighted RMSE loss
    """

    def __init__(
        self,
        res,
        data_dir,
        aux_data_dir,
        weight_per_variable=False,
        start_ind=0,
        end_ind=24,
    ):

        super().__init__()
        self.start_ind = start_ind
        self.end_ind = end_ind

        self.weights = torch.from_numpy(
            np.load(aux_data_dir + "lat_weights/weights_lat_{}.npy".format(res))[
                np.newaxis, ..., np.newaxis
            ]
        ).float()

        self.weight_per_variable = weight_per_variable
        self.variable_weights = torch.from_numpy(
            np.load(aux_data_dir + "loss_weights.npy")[
                np.newaxis, np.newaxis, np.newaxis, start_ind:end_ind
            ]
        ).float()

    def forward(
        self,
        target,
        output,
        prev_step_output,
        fix_sigma=False,
        unwrap=False,
        expand=False,
    ):
        squared_diff = (target.to(output.device) - output) ** 2

        if not expand:
            weighted_sqared_diff = squared_diff * self.weights.to(target.device)

            if self.weight_per_variable:
                weighted_sqared_diff = weighted_sqared_diff * self.variable_weights.to(
                    weighted_sqared_diff.device
                )

            x = torch.nanmean(
                torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2, 3)))
            )

            return x

        weighted_sqared_diff = squared_diff * self.weights.to(target.device)
        x = torch.mean(
            torch.sqrt(torch.nanmean(weighted_sqared_diff, dim=(1, 2))), dim=0
        )

        return x


class WeightedCRPSLoss(nn.Module):
    """
    Latitude weighted CRPS loss (Energy Score).

    Always applies latitude weights. Optionally applies variable weights
    (weight_per_variable=True) and pressure level weights
    (weight_per_pressure=True, requires data_dir and era5_mode).
    """

    def __init__(
        self,
        res,
        aux_data_dir,
        data_dir=None,
        era5_mode=None,
        weight_per_variable=False,
        weight_per_pressure=False,
        start_ind=0,
        end_ind=24,
    ):
        super().__init__()
        self.start_ind = start_ind
        self.end_ind = end_ind

        self.weights = torch.from_numpy(
            np.load(aux_data_dir + "lat_weights/weights_lat_{}.npy".format(res))[
                np.newaxis, ..., np.newaxis
            ]
        ).float()

        self.weight_per_variable = weight_per_variable
        self.variable_weights = torch.from_numpy(
            np.load(aux_data_dir + "loss_weights.npy")[
                np.newaxis, np.newaxis, np.newaxis, start_ind:end_ind
            ]
        ).float()

        self.weight_per_pressure = weight_per_pressure
        if weight_per_pressure:
            assert data_dir is not None and era5_mode is not None
            pressure_file = (
                "era5/era5_pressure_levels_{}_24ch.npy".format(era5_mode)
                if era5_mode == "4u"
                else "era5/era5_pressure_levels_{}.npy".format(era5_mode)
            )
            self.pressure_levels = (
                torch.from_numpy(
                    np.load(data_dir + pressure_file)[np.newaxis, np.newaxis, np.newaxis, :]
                ).float()
                / 1000
            )
        else:
            self.pressure_levels = None

    def forward(
        self,
        target,
        output,
        prev_step_output=None,
        fix_sigma=False,
        unwrap=False,
        expand=False,
    ):
        target = target.to(output.device)
        if output.ndim == 5:
            # Ensemble prediction (B, M, H, W, C): compute Energy Score / CRPS
            target_expanded = target.unsqueeze(1)
            mae_term = torch.nanmean(torch.abs(output - target_expanded), dim=1)
            diff_matrix = torch.abs(output.unsqueeze(2) - output.unsqueeze(1))
            spread_term = 0.5 * torch.nanmean(diff_matrix, dim=(1, 2))
            loss_map = mae_term - spread_term  # (B, H, W, C)
        else:
            loss_map = torch.abs(output - target)  # (B, H, W, C)

        weighted_loss = loss_map * self.weights.to(loss_map.device)

        if self.weight_per_variable:
            weighted_loss = weighted_loss * self.variable_weights.to(weighted_loss.device)

        if self.weight_per_pressure:
            weighted_loss = weighted_loss * self.pressure_levels.to(weighted_loss.device)

        if expand:
            return torch.nanmean(weighted_loss, dim=(0, 1, 2))
        else:
            return torch.nanmean(weighted_loss)


class DownscalingCRPSLoss(nn.Module):
    """
    CRPS loss (Energy Score) for station-level predictions.
    Handles NaN targets. When output has an ensemble dimension (B, M, S),
    computes CRPS. Falls back to MAE for deterministic (B, S) output.
    """

    def __init__(self):
        super().__init__()

    def forward(self, target, output, prev_step=None, fix_sigma=None, expand=False):
        target = target.to(output.device)

        if output.ndim == 3:
            B, M, S = output.shape
            target_flat = target.reshape(B, S)

            valid = ~torch.isnan(target_flat)
            if valid.sum() == 0:
                return torch.tensor(float("nan"), device=output.device, requires_grad=True)

            target_clean = torch.where(valid, target_flat, torch.zeros_like(target_flat))
            target_exp = target_clean.unsqueeze(1)  # (B, 1, S)

            abs_diff = torch.abs(output - target_exp) * valid.unsqueeze(1).float()
            mae_term = abs_diff.sum(dim=1) / M

            diff_matrix = torch.abs(output.unsqueeze(2) - output.unsqueeze(1))
            diff_matrix = diff_matrix * valid.unsqueeze(1).unsqueeze(1).float()
            spread_term = 0.5 * diff_matrix.sum(dim=(1, 2)) / (M * M)

            crps = (mae_term - spread_term) * valid.float()
            return crps.sum() / valid.sum()

        else:
            target_flat = torch.flatten(target)
            output_flat = torch.flatten(output)
            valid = ~torch.isnan(target_flat)
            if valid.sum() == 0:
                return torch.tensor(float("nan"), device=output.device, requires_grad=True)
            return torch.mean(torch.abs(target_flat[valid] - output_flat[valid]))
