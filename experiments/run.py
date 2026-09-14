import os
import json
import random
import numpy as np
import torch

from config import Config
import methods.base_method as method

import models.base_gru_chunked as exp_chunked
import models.base_gru_full as exp_full


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(exp_name, model_module, cfg):
    print(f"\nСТАРТ ЭКСПЕРИМЕНТА: {exp_name}")

    train_loader = model_module.get_dataloader(cfg)
    exp_results = []

    for i, seed in enumerate(cfg.seeds, 1):
        print(f"[{i}/{len(cfg.seeds)}] Сид: {seed} | Эксперимент: {exp_name}")
        set_seed(seed)

        model = model_module.create_model(cfg)
        final_full_wp = method.train_seed(model, train_loader, cfg, exp_name, seed)
        exp_results.append({"seed": seed, "final_full_wp": final_full_wp})

    scores = [r["final_full_wp"] for r in exp_results]
    mean_wp = float(np.mean(scores))
    std_wp = float(np.std(scores))

    exp_dir = os.path.join(cfg.runs_dir, exp_name)
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
        # ("base_gru_chunked", exp_chunked),
        ("base_gru_full", exp_full),
    ]

    final_comparison = {}

    for exp_name, model_module in experiments:
        mean_wp, std_wp = run_experiment(exp_name, model_module, cfg)
        final_comparison[exp_name] = f"{mean_wp:.5f} ± {std_wp:.5f}"

    print("\nИТОГОВОЕ СРАВНЕНИЕ:")
    for exp_name, result in final_comparison.items():
        print(f"{exp_name}: {result}")


if __name__ == "__main__":
    main()