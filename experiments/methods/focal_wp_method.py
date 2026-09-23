from methods.trainer import generic_train_seed
from methods.losses import focal_weighted_pearson_loss

def train_seed(model, train_loader, cfg, exp_name, seed):
    gamma = getattr(cfg, "focal_gamma", 1.0)
    loss_fn = lambda p, y: focal_weighted_pearson_loss(p, y, gamma=gamma)
    return generic_train_seed(model, train_loader, cfg, exp_name, seed, loss_fn=loss_fn, scheduler_type="cosine")