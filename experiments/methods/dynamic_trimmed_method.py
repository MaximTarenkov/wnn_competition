from methods.trainer import generic_train_seed
from methods.losses import dynamic_trimmed_loss


def train_seed(model, train_loader, cfg, exp_name, seed):
    loss_fn = lambda p, y: dynamic_trimmed_loss(p, y, keep_ratio=0.70, soft_weight=0.05)
    return generic_train_seed(
        model, train_loader, cfg, exp_name, seed,
        loss_fn=loss_fn,
        scheduler_type="cosine"
    )