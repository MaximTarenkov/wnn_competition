import os
import json
import random
import numpy as np
import torch

from config import Config

import methods.base_method as base_method
import methods.local_loss as local_loss
import methods.local_global_loss as local_global_loss
import methods.local_global_loss_7_3 as local_global_loss_7_3

from models import gru_mlp_encoders
from models import gru_gated_input
from models import gru_gated_output
from models import vgru_chunked
from models import gru_mlp_encoders_sort
from models import gru_mlp_encoders_l1
from models import gru_chrono_init
import models.base_gru_chunked as exp_chunked
import models.base_gru_full as exp_full


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_experiment(exp_name, model_module, method_module, cfg):
    print(f"СТАРТ ЭКСПЕРИМЕНТА: {exp_name}")

    exp_results = []

    for i, seed in enumerate(cfg.seeds, 1):
        print(f"[{i}/{len(cfg.seeds)}] Сид: {seed} | Эксперимент: {exp_name}")
        set_seed(seed)

        model = model_module.create_model(cfg)
        train_loader = model_module.get_dataloader(cfg, seed=seed)

        final_full_wp = method_module.train_seed(model, train_loader, cfg, exp_name, seed)
        exp_results.append({"seed": seed, "final_full_wp": final_full_wp})

    scores = [r["final_full_wp"] for r in exp_results]
    mean_wp = float(np.mean(scores))
    std_wp = float(np.std(scores))

    exp_dir = os.path.join(cfg.runs_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    summary_path = os.path.join(exp_dir, "experiment_summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": exp_name,
            "mean_final_full_wp": mean_wp,
            "std_final_full_wp": std_wp,
            "details": exp_results
        }, f, indent=4)

    print(f"ИТОГ [{exp_name}]: {mean_wp:.5f} ± {std_wp:.5f}")
    return mean_wp, std_wp


def main():
    cfg = Config()

    experiments = [
        #("gru_base_loss", exp_chunked, base_method),
        #("gru_local_loss", exp_chunked, local_loss),
        #("gru_local-global_loss", exp_chunked, local_global_loss),
        #("gru_local-global_loss_7_3", exp_chunked, local_global_loss_7_3)
        #("gru_mlp_encoders", gru_mlp_encoders, base_method),
        #("gru_gated_input", gru_gated_input, base_method),
        #("gru_gated_output", gru_gated_output, base_method),
        #("vgru_chunked", vgru_chunked, base_method),
        #("gru_mlp_encoders_sortonly", gru_mlp_encoders_sort, base_method),
        ("gru_chrono_init", gru_chrono_init, base_method),


    ]

    final_comparison = {}

    for exp_name, model_module, method_module in experiments:
        mean_wp, std_wp = run_experiment(exp_name, model_module, method_module, cfg)
        final_comparison[exp_name] = f"{mean_wp:.5f} ± {std_wp:.5f}"

    print("\nИТОГОВОЕ СРАВНЕНИЕ:")
    for exp_name, result in final_comparison.items():
        print(f"{exp_name}: {result}")


if __name__ == "__main__":
    main()