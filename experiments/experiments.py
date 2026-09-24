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
    gru_mlp_encoders_disentangled_micro_highway,
    gru_mlp_encoders_disentangled_advanced_feats,
    gru_mlp_encoders_disentangled_two_heads,
    gru_mlp_encoders_disentangled_l1_dualstream_delta_silu,
    gru_mlp_encoders_disentangled_l1_delta_expanded,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_v2,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_resgru,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_skip,
    gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_sigm,
    gru_input_encoders,
    gru_input_encoders_micro,
    gru_input_encoders_l1,
    gru_input_encoders_l1_delta,
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


def base_gru_5b():
    return {
        "name": "base_gru_5b",
        "model": base_gru,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def base_gru_wd1e2_5b():
    return {
        "name": "base_gru_wd1e2_5b",
        "model": base_gru,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def base_gru_h90_5b():
    return {
        "name": "base_gru_h90_5b",
        "model": base_gru,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 90,
        "batch_size": 5,
    }


def exp_gru_input_encoders():
    return {
        "name": "gru_input_encoders_ep4",
        "model": gru_input_encoders,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_micro():
    return {
        "name": "gru_input_encoders_micro_ep4",
        "model": gru_input_encoders_micro,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_l1():
    return {
        "name": "gru_input_encoders_l1_ep4",
        "model": gru_input_encoders_l1,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }

def exp_gru_input_encoders_l1_delta():
    return {
        "name": "gru_input_encoders_l1_delta_ep4",
        "model": gru_input_encoders_l1_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }

