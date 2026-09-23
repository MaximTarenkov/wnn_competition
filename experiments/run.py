import os
import random
import numpy as np
import torch
import gc

from config import Config

import methods.base_method as base_method
import methods.local_global_loss_7_3 as local_global_loss_7_3
from methods import (
    asym_wp_loss,
    cosine_scheduler_method,
    swa_method,
    mse_anchor,
    dynamic_trimmed_method,
    focal_wp_method
)

from models import (
    base_gru,
    gru_mlp_encoders,
    gru_gated_input,
    gru_gated_output,
    vgru_chunked,
    gru_mlp_encoders_sort,
    gru_mlp_encoders_l1,
    gru_chrono_init,
    base_gru_asym_tanh,
    gru_disentangled_encoders_l1,
    gru_sum_diff,
    gru_mlp_encoders_disentangled_micro,
    gru_mlp_encoders_disentangled_micro_highway,
    gru_mlp_encoders_disentangled_l1_fix,
    gru_mlp_encoders_disentangled_advanced_feats,
    gru_mlp_encoders_disentangled_two_heads,
    gru_mlp_encoders_disentangled_l1_delta_silu,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix,
    gru_mlp_encoders_disentangled_l1_dualstream_delta_silu,
    gru_mlp_encoders_disentangled_l1_delta_expanded,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_v2,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_resgru,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_skip,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_sigm,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        train_loader = dataset.get_train_dataloader(cfg, seed=seed)

        final_full_wp = method_module.train_seed(model, train_loader, cfg, exp_name, seed)
        exp_results.append({"seed": seed, "final_full_wp": final_full_wp})

        del model, train_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    scores = [r["final_full_wp"] for r in exp_results]
    mean_wp = float(np.mean(scores))
    std_wp = float(np.std(scores))

    exp_dir = os.path.join(cfg.runs_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"ИТОГ [{exp_name}]: {mean_wp:.5f} ± {std_wp:.5f}")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mean_wp, std_wp


def main():
    cfg = Config()

    experiments = [
        # ("base_gru", base_gru, base_method),
        # ("base_gru_12b", base_gru, base_method),
        # ("base_gru__6)", base_gru, base_method),
        # ("base_gru__local_loss", base_gru, local_loss),
        # ("base_gru__local_global_50_50", base_gru, local_global_loss),
        # ("base_gru__local_global_70_30", base_gru, local_global_loss_7_3),
        # ("base_gru__local_global_70_30_6b", base_gru, local_global_loss_7_3),
        # ("gru_chrono_init", gru_chrono_init, base_method),
        # ("gru_gated_input_6b", gru_gated_input, base_method),  
        # ("gru_gated_output", gru_gated_output, base_method),
        # ("gru_mlp_encoders_6b", gru_mlp_encoders, base_method),
        # ("gru_mlp_encoders_sort", gru_mlp_encoders_sort, base_method),
        # ("gru_mlp_encoders_l1_6b", gru_mlp_encoders_l1, base_method),
        # ("vgru_chunked", vgru_chunked, base_method),
        # ("gru_mlp_encoders_disentangled_l1_6b", gru_disentangled_encoders_l1, base_method),
        # ("gru_sum_diff_6b", gru_sum_diff, base_method),
        # ("gru_mlp_encoders_disentangled_micro_6b", gru_mlp_encoders_disentangled_micro, base_method),
        # ("gru_mlp_encoders_disentangled_micro_highway_6b", gru_mlp_encoders_disentangled_micro_highway, base_method),
        # ("gru_mlp_encoders_disentangled_micro_coslr_6b", gru_mlp_encoders_disentangled_micro, cosine_scheduler_method),
        # ("gru_mlp_encoders_disentangled_micro_swa_6b", gru_mlp_encoders_disentangled_micro, swa_method),
        # ("gru_mlp_encoders_disentangled_l1_fix_6b", gru_mlp_encoders_disentangled_l1_fix, base_method),
        # ("gru_mlp_encoders_disentangled_advanced_feats_6b", gru_mlp_encoders_disentangled_advanced_feats, base_method),
        # ("gru_mlp_encoders_disentangled_two_heads_6b", gru_mlp_encoders_disentangled_two_heads, base_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_6b", gru_mlp_encoders_disentangled_l1_delta_silu, base_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_6b", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix, base_method),
        # ("gru_mlp_encoders_disentangled_l1_dualstream_delta_silu_6b", gru_mlp_encoders_disentangled_l1_dualstream_delta_silu, base_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_expanded_6b", gru_mlp_encoders_disentangled_l1_delta_expanded, base_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_v2_6b", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_v2, base_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_8b", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix, base_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_resgru", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_resgru, base_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_5b", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix, cosine_scheduler_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_skip_coslr_5b", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_skip, cosine_scheduler_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_sigm_coslr_5b", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_sigm, cosine_scheduler_method),
        # ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr-mse_5b", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix, mse_anchor),
        ("gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_focal_5b_2g", gru_mlp_encoders_disentangled_l1_delta_silu_hotfix, focal_wp_method),
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