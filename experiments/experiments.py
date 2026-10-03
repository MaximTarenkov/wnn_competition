from models import (
    base_gru,
    gru_gated_input_encoders_delta,
    gru_gated_input,
    gru_gated_output,
    gru_input_encoders_delta_gated_skip,
    gru_input_encoders_delta_highway,
    gru_input_encoders_delta_sigm,
    gru_input_encoders_delta_two_heads,
    gru_input_encoders_delta,
    gru_input_encoders_l1_fix_delta,
    gru_input_encoders_vol_agg_delta,
    gru_input_encoders,
    gru_sum_diff,
    vgru_chunked,
    base_gru_delta,
    base_gru_vol_agg_delta,
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


def exp_gru_input_encoders():
    return {
        "name": "gru_input_encoders",
        "model": gru_input_encoders,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_delta():
    return {
        "name": "gru_input_encoders_delta",
        "model": gru_input_encoders_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_l1_fix_delta():
    return {
        "name": "gru_input_encoders_l1_fix_delta",
        "model": gru_input_encoders_l1_fix_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_vol_agg_delta():
    return {
        "name": "gru_input_encoders_vol_agg_delta",
        "model": gru_input_encoders_vol_agg_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_gated_input_encoders_delta():
    return {
        "name": "gru_gated_input_encoders_delta",
        "model": gru_gated_input_encoders_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_delta_gated_skip():
    return {
        "name": "gru_input_encoders_delta_gated_skip",
        "model": gru_input_encoders_delta_gated_skip,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_delta_highway():
    return {
        "name": "gru_input_encoders_delta_highway",
        "model": gru_input_encoders_delta_highway,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_delta_sigm():
    return {
        "name": "gru_input_encoders_delta_sigm",
        "model": gru_input_encoders_delta_sigm,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_delta_two_heads():
    return {
        "name": "gru_input_encoders_delta_two_heads",
        "model": gru_input_encoders_delta_two_heads,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_gated_input():
    return {
        "name": "gru_gated_input",
        "model": gru_gated_input,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_gated_output():
    return {
        "name": "gru_gated_output",
        "model": gru_gated_output,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_sum_diff():
    return {
        "name": "gru_sum_diff",
        "model": gru_sum_diff,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_vgru_chunked():
    return {
        "name": "vgru_chunked",
        "model": vgru_chunked,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }

def exp_base_gru_delta():
    return {
        "name": "base_gru_delta",
        "model": base_gru_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_vbase_gru_vol_agg_delta():
    return {
        "name": "base_gru_vol_agg_delta",
        "model": base_gru_vol_agg_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }

def base_gru_640b():
    return {
        "name": "base_gru_640b",
        "model": base_gru,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 640,
    }