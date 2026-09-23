from models import (
    base_gru,
    gru_chrono_init,
    gru_gated_input,
    gru_gated_output,
    gru_mlp_encoders,
    gru_mlp_encoders_sort,
    gru_mlp_encoders_l1,
    vgru_chunked,
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

from methods import (
    base_method,
    local_global_loss_7_3,
    cosine_scheduler_method,
    swa_method,
    mse_anchor,
    focal_wp_method,
    dynamic_trimmed_method,
    asym_wp_loss,
)


def exp_base_gru_lr1e4():
    return {
        "name": "base_gru_lr1e4",
        "model": base_gru,
        "method": base_method,
        "lr": 1e-4,
    }

def exp_base_gru_lr1e5():
    return {
        "name": "base_gru_lr1e5",
        "model": base_gru,
        "method": base_method,
        "lr": 1e-5,
    }

def exp_base_gru_lr1e3():
    return {
        "name": "base_gru_lr1e3",
        "model": base_gru,
        "method": base_method,
        "lr": 1e-3,
    }


def exp_base_gru_12b():
    return {
        "name": "base_gru_12b",
        "model": base_gru,
        "method": base_method,
        "batch_size": 12,
    }


def exp_base_gru_6b():
    return {
        "name": "base_gru_6b",
        "model": base_gru,
        "method": base_method,
        "batch_size": 6,
    }


def exp_base_gru_local_global_70_30():
    return {
        "name": "base_gru__local_global_70_30",
        "model": base_gru,
        "method": local_global_loss_7_3,
    }


def exp_base_gru_local_global_70_30_6b():
    return {
        "name": "base_gru__local_global_70_30_6b",
        "model": base_gru,
        "method": local_global_loss_7_3,
        "batch_size": 6,
    }


def exp_gru_chrono_init():
    return {
        "name": "gru_chrono_init",
        "model": gru_chrono_init,
        "method": base_method,
    }


def exp_gru_gated_input_6b():
    return {
        "name": "gru_gated_input_6b",
        "model": gru_gated_input,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_gated_output():
    return {
        "name": "gru_gated_output",
        "model": gru_gated_output,
        "method": base_method,
    }


def exp_gru_mlp_encoders_6b():
    return {
        "name": "gru_mlp_encoders_6b",
        "model": gru_mlp_encoders,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_sort():
    return {
        "name": "gru_mlp_encoders_sort",
        "model": gru_mlp_encoders_sort,
        "method": base_method,
    }


def exp_gru_mlp_encoders_l1_6b():
    return {
        "name": "gru_mlp_encoders_l1_6b",
        "model": gru_mlp_encoders_l1,
        "method": base_method,
        "batch_size": 6,
    }


def exp_vgru_chunked():
    return {
        "name": "vgru_chunked",
        "model": vgru_chunked,
        "method": base_method,
    }


def exp_gru_mlp_encoders_disentangled_l1_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_6b",
        "model": gru_disentangled_encoders_l1,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_sum_diff_6b():
    return {
        "name": "gru_sum_diff_6b",
        "model": gru_sum_diff,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_micro_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_micro_6b",
        "model": gru_mlp_encoders_disentangled_micro,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_micro_highway_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_micro_highway_6b",
        "model": gru_mlp_encoders_disentangled_micro_highway,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_micro_coslr_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_micro_coslr_6b",
        "model": gru_mlp_encoders_disentangled_micro,
        "method": cosine_scheduler_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_micro_swa_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_micro_swa_6b",
        "model": gru_mlp_encoders_disentangled_micro,
        "method": swa_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_l1_fix_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_fix_6b",
        "model": gru_mlp_encoders_disentangled_l1_fix,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_advanced_feats_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_advanced_feats_6b",
        "model": gru_mlp_encoders_disentangled_advanced_feats,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_two_heads_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_two_heads_6b",
        "model": gru_mlp_encoders_disentangled_two_heads,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_6b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_6b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_l1_dualstream_delta_silu_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_dualstream_delta_silu_6b",
        "model": gru_mlp_encoders_disentangled_l1_dualstream_delta_silu,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_expanded_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_expanded_6b",
        "model": gru_mlp_encoders_disentangled_l1_delta_expanded,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_v2_6b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_v2_6b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_v2,
        "method": base_method,
        "batch_size": 6,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_8b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_8b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix,
        "method": base_method,
        "batch_size": 8,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_resgru():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_resgru",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_resgru,
        "method": base_method,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_5b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_5b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix,
        "method": cosine_scheduler_method,
        "batch_size": 5,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_skip_coslr_5b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_skip_coslr_5b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_skip,
        "method": cosine_scheduler_method,
        "batch_size": 5,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_sigm_coslr_5b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_sigm_coslr_5b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_sigm,
        "method": cosine_scheduler_method,
        "batch_size": 5,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_mse_5b():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr-mse_5b",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix,
        "method": mse_anchor,
        "batch_size": 5,
        "lambda_anchor": 0.05,
    }


def exp_gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_focal_5b_2g():
    return {
        "name": "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_focal_5b_2g",
        "model": gru_mlp_encoders_disentangled_l1_delta_silu_hotfix,
        "method": focal_wp_method,
        "batch_size": 5,
        "focal_gamma": 2.0,
    }