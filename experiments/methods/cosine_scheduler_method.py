from methods.trainer import generic_train_seed
from methods.losses import base_weighted_pearson_loss

def train_seed(model, train_loader, cfg, exp_name, seed):
    return generic_train_seed(model, train_loader, cfg, exp_name, seed, loss_fn=base_weighted_pearson_loss, scheduler_type="cosine")