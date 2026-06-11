"""
NHITs model builder
"""
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss

from utils import H, VAL_CHECK, EARLY_STOP


def build_nhits_model(
    input_size,
    n_stacks,
    mlp_units,
    learning_rate,
    batch_size,
    max_steps,
    features,
    seed=42,
    early_stop=EARLY_STOP,
):
    """
    Build and return a configured NHITs model.

    Parameters
    ----------
    input_size, n_stacks, mlp_units, learning_rate, batch_size, max_steps :
        hyperparameters (from Optuna)
    features   : list of exogenous covariate column names
    seed       : random seed
    early_stop : early stopping patience in steps
    """
    pool_kernels = [4, 2, 1][:n_stacks]
    freq_down    = [4, 2, 1][:n_stacks]

    return NHITS(
        h                         = H,
        input_size                = input_size,
        futr_exog_list            = features,
        loss                      = DistributionLoss(
                                        distribution='StudentT',
                                        level=[50, 80, 90, 95],
                                    ),
        n_blocks                  = [1] * n_stacks,
        mlp_units                 = [[mlp_units, mlp_units]] * n_stacks,
        n_pool_kernel_size        = pool_kernels,
        n_freq_downsample         = freq_down,
        dropout_prob_theta        = 0.0,
        max_steps                 = max_steps,
        batch_size                = batch_size,
        learning_rate             = learning_rate,
        val_check_steps           = VAL_CHECK,
        early_stop_patience_steps = early_stop,
        scaler_type               = 'identity',
        random_seed               = seed,
        enable_progress_bar       = False,
        enable_model_summary      = False,
        logger                    = False,
        accelerator               = 'cpu',
        devices                   = 1,
    )
