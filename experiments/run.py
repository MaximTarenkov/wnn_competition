import os
import random
import numpy as np
import torch
import gc

from rich.console import Console
from rich.table import Table

from config import Config
import dataset
import methods.trainer as trainer
from experiments import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(experiments_group, cfg):
    console = Console()

    if isinstance(experiments_group, dict):
        experiments_group = [experiments_group]

    exp_names = [e["name"] for e in experiments_group]

    console.print(f"\n[bold green]STARTING RUN FOR {len(experiments_group)} EXPERIMENT(S) ON SHARED STREAM:[/bold green]")
    for name in exp_names:
        console.print(f"  [cyan]•[/cyan] {name}")

    results_by_exp = {name: [] for name in exp_names}

    for i, seed in enumerate(cfg.seeds, 1):
        console.print(f"[bold blue][{i}/{len(cfg.seeds)}] Seed: {seed} | Running parallel stream...[/bold blue]")
        set_seed(seed)

        cfg.full_batch_size = max(e.get("batch_size", cfg.full_batch_size) for e in experiments_group)

        train_loader = dataset.get_train_dataloader(cfg, seed=seed)

        seed_scores = trainer.train_seed(experiments_group, train_loader, cfg, seed=seed)

        if isinstance(seed_scores, dict):
            for name, score in seed_scores.items():
                results_by_exp[name].append(score)
        else:
            results_by_exp[exp_names[0]].append(seed_scores)

        del train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    table = Table(title="\nFINAL COMPARISON (MEAN ± STD ACROSS SEEDS)", show_header=True, header_style="bold magenta")
    table.add_column("Experiment Name", style="cyan", width=45)
    table.add_column("Full Val WP (Mean ± Std)", style="bold green", justify="right")

    for name, scores in results_by_exp.items():
        mean_wp = float(np.mean(scores))
        std_wp = float(np.std(scores))
        table.add_row(name, f"{mean_wp:.5f} ± {std_wp:.5f}")

    console.print(table)
    return results_by_exp


def main():
    cfg = Config()

    active_experiments = [
        base_gru_5b(),
        exp_gru_input_encoders_micro_delta(),         
        exp_gru_dualstream_l1_delta(),   
        exp_gru_dualstream_micro_delta(),
    ]

    run_experiment(active_experiments, cfg)


if __name__ == "__main__":
    main()
