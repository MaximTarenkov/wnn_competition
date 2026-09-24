import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import gc

from rich.console import Console
from rich.table import Table

from config import Config
import dataset
import methods.trainer as trainer
from experiments import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_experiment_task(exp_fn, cfg):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    exp_dict = exp_fn()
    exp_name = exp_dict["name"]
    results = []

    for seed in cfg.seeds:
        set_seed(seed)
        train_loader = dataset.get_train_dataloader(cfg, seed=seed)
        score = trainer.train_seed(exp_dict, train_loader, cfg, exp_name=exp_name, seed=seed)
        results.append(score)
        del train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return exp_name, results


def main():
    mp.set_start_method("spawn", force=True)

    cfg = Config()
    console = Console()

    active_experiment_fns = [
        exp_gru_dualstream_micro_delta,
        exp_gru_input_encoders_micro_delta_skip,
        exp_gru_input_encoders_micro_delta_two_heads,
        exp_gru_input_encoders_micro_delta_sigm,
        exp_gru_input_encoders_micro_delta_resgru,
        exp_gru_input_encoders_micro_delta,
        exp_gru_input_encoders_micro_delta_highway,
        exp_gru_dualstream_l1_delta,
        exp_gru_input_encoders_l1_delta_6tan,
        exp_gru_sum_diff,
        exp_base_gru_asym_tanh,
        exp_gru_gated_input,
        exp_gru_gated_output,
        exp_gru_chrono_init,
        exp_vgru_chunked,
    ]

    max_workers = 4
    console.print(f"\n[bold green]STARTING PARALLEL POOL FOR {len(active_experiment_fns)} EXPERIMENTS ({max_workers} WORKERS):[/bold green]")

    results_by_exp = {}
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(run_single_experiment_task, exp_fn, cfg) for exp_fn in active_experiment_fns]
        for f in futures:
            name, scores = f.result()
            results_by_exp[name] = scores
            console.print(f"  [bold green]✓ COMPLETED:[/bold green] [cyan]{name:<35}[/cyan] -> WP: {np.mean(scores):.5f}")

    table = Table(title="\nFINAL COMPARISON (MEAN ± STD ACROSS SEEDS)", show_header=True, header_style="bold magenta")
    table.add_column("Experiment Name", style="cyan", width=45)
    table.add_column("Full Val WP (Mean ± Std)", style="bold green", justify="right")

    for name, scores in results_by_exp.items():
        mean_wp = float(np.mean(scores))
        std_wp = float(np.std(scores))
        table.add_row(name, f"{mean_wp:.5f} ± {std_wp:.5f}")

    console.print(table)


if __name__ == "__main__":
    main()