from models import (
    base_gru,
    base_gru_asym_tanh,
    gru_chrono_init,
    gru_gated_input,
    gru_gated_output,
    gru_input_encoders,
    gru_input_encoders_micro,
    gru_input_encoders_l1,
    gru_input_encoders_l1_delta,
    gru_input_encoders_l1_delta_6tan,
    gru_input_encoders_micro_delta,
    gru_input_encoders_micro_delta_highway,
    gru_input_encoders_micro_delta_resgru,
    gru_input_encoders_micro_delta_sigm,
    gru_input_encoders_micro_delta_skip,
    gru_input_encoders_micro_delta_two_heads,
    gru_dualstream_l1_delta,
    gru_dualstream_micro_delta,
    gru_sum_diff,
    vgru_chunked,
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
        "name": "gru_input_encoders_ep4",
        "model": gru_input_encoders,
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


def exp_gru_input_encoders_l1_delta_6tan():
    return {
        "name": "gru_input_encoders_l1_delta_6tan",
        "model": gru_input_encoders_l1_delta_6tan,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_micro_delta():
    return {
        "name": "gru_input_encoders_micro_delta",
        "model": gru_input_encoders_micro_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_micro_delta_skip():
    return {
        "name": "gru_input_encoders_micro_delta_skip",
        "model": gru_input_encoders_micro_delta_skip,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_micro_delta_two_heads():
    return {
        "name": "gru_input_encoders_micro_delta_two_heads",
        "model": gru_input_encoders_micro_delta_two_heads,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_micro_delta_sigm():
    return {
        "name": "gru_input_encoders_micro_delta_sigm",
        "model": gru_input_encoders_micro_delta_sigm,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_micro_delta_resgru():
    return {
        "name": "gru_input_encoders_micro_delta_resgru",
        "model": gru_input_encoders_micro_delta_resgru,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_input_encoders_micro_delta_highway():
    return {
        "name": "gru_input_encoders_micro_delta_highway",
        "model": gru_input_encoders_micro_delta_highway,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_dualstream_l1_delta():
    return {
        "name": "gru_dualstream_l1_delta",
        "model": gru_dualstream_l1_delta,
        "method": cosine_scheduler_method,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "hidden_dim": 128,
        "batch_size": 5,
    }


def exp_gru_dualstream_micro_delta():
    return {
        "name": "gru_dualstream_micro_delta",
        "model": gru_dualstream_micro_delta,
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


def exp_base_gru_asym_tanh():
    return {
        "name": "base_gru_asym_tanh",
        "model": base_gru_asym_tanh,
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


def exp_gru_chrono_init():
    return {
        "name": "gru_chrono_init",
        "model": gru_chrono_init,
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