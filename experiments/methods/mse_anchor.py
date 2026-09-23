import math
from methods.trainer import generic_train_seed
from methods.losses import mse_anchor_loss

def anchor_step_hook(global_step, total_steps, cfg):
    lambda_init = getattr(cfg, "lambda_anchor", 0.05)
    progress = min(1.0, global_step / total_steps)
    lambda_t = 0.5 * lambda_init * (1.0 + math.cos(math.pi * progress))
    return {"lambda_anchor": lambda_t}

def train_seed(model, train_loader, cfg, exp_name, seed):
    return generic_train_seed(
        model, train_loader, cfg, exp_name, seed,
        loss_fn=mse_anchor_loss,
        scheduler_type="cosine",
        loss_step_hook=anchor_step_hook
    )