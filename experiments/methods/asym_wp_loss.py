from methods.trainer import generic_train_seed
from methods.losses import asym_corr_penalty_loss


def train_seed(model, train_loader, cfg, exp_name, seed):
    return generic_train_seed(
        model, train_loader, cfg, exp_name, seed,
        loss_fn=asym_corr_penalty_loss,
        scheduler_type=None
    )