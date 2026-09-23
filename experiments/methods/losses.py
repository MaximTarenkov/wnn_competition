import torch
import torch.nn.functional as F

METRIC_CLIP = 2.0


def base_weighted_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    p_flat = pred.reshape(-1, 2)
    t_flat = torch.clamp(target.reshape(-1, 2), -METRIC_CLIP, METRIC_CLIP)
    w = torch.abs(t_flat).clamp(min=eps)

    loss = 0.0
    for i in range(2):
        p, t, wi = p_flat[:, i], t_flat[:, i], w[:, i]
        w_sum = torch.sum(wi)
        p_mean = torch.sum(wi * p) / w_sum
        t_mean = torch.sum(wi * t) / w_sum
        p_diff, t_diff = p - p_mean, t - t_mean

        cov = torch.sum(wi * p_diff * t_diff) / w_sum
        p_var = torch.sum(wi * p_diff ** 2) / w_sum
        t_var = torch.sum(wi * t_diff ** 2) / w_sum

        corr = cov / (torch.sqrt(p_var + eps) * torch.sqrt(t_var + eps) + eps)
        loss = loss - corr

    return loss / 2.0


def focal_weighted_pearson_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    p_flat = pred.reshape(-1, 2)
    t_flat = torch.clamp(target.reshape(-1, 2), -METRIC_CLIP, METRIC_CLIP)
    w_base = torch.abs(t_flat).clamp(min=eps)

    abs_err = torch.abs(p_flat - t_flat).detach()
    focal_weights = w_base * (1.0 + gamma * abs_err)

    loss = 0.0
    for i in range(2):
        p, t, w = p_flat[:, i], t_flat[:, i], focal_weights[:, i]
        w_sum = torch.sum(w)
        p_mean = torch.sum(w * p) / w_sum
        t_mean = torch.sum(w * t) / w_sum
        p_diff, t_diff = p - p_mean, t - t_mean

        cov = torch.sum(w * p_diff * t_diff) / w_sum
        p_var = torch.sum(w * p_diff ** 2) / w_sum
        t_var = torch.sum(w * t_diff ** 2) / w_sum

        corr = cov / (torch.sqrt(p_var + eps) * torch.sqrt(t_var + eps) + eps)
        loss = loss - corr

    return loss / 2.0


def dynamic_trimmed_loss(pred: torch.Tensor, target: torch.Tensor, keep_ratio: float = 0.70, soft_weight: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    p_flat = pred.reshape(-1, 2)
    t_flat = torch.clamp(target.reshape(-1, 2), -METRIC_CLIP, METRIC_CLIP)
    w_raw = torch.abs(t_flat).clamp(min=eps)
    point_err = w_raw * (p_flat - t_flat) ** 2

    loss = 0.0
    for i in range(2):
        err_i = point_err[:, i]
        k = int(err_i.size(0) * keep_ratio)
        threshold = torch.kthvalue(err_i, k).values.detach()
        mask_easy = (err_i <= threshold).float()
        w = w_raw[:, i] * (mask_easy + soft_weight * (1.0 - mask_easy))

        w_sum = torch.sum(w)
        p_mean = torch.sum(w * p_flat[:, i]) / w_sum
        t_mean = torch.sum(w * t_flat[:, i]) / w_sum
        p_diff, t_diff = p_flat[:, i] - p_mean, t_flat[:, i] - t_mean

        cov = torch.sum(w * p_diff * t_diff) / w_sum
        p_var = torch.sum(w * p_diff ** 2) / w_sum
        t_var = torch.sum(w * t_diff ** 2) / w_sum

        corr = cov / (torch.sqrt(p_var + eps) * torch.sqrt(t_var + eps) + eps)
        loss = loss - corr

    return loss / 2.0


def mse_anchor_loss(pred: torch.Tensor, target: torch.Tensor, lambda_anchor: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    p_flat = pred.reshape(-1, 2)
    t_flat = torch.clamp(target.reshape(-1, 2), -METRIC_CLIP, METRIC_CLIP)
    weights = torch.abs(t_flat).clamp(min=eps)

    loss = 0.0
    for i in range(2):
        p, t, w = p_flat[:, i], t_flat[:, i], weights[:, i]
        w_sum = torch.sum(w)
        p_mean = torch.sum(w * p) / w_sum
        t_mean = torch.sum(w * t) / w_sum
        p_diff, t_diff = p - p_mean, t - t_mean

        cov = torch.sum(w * p_diff * t_diff) / w_sum
        p_var = torch.sum(w * p_diff ** 2) / w_sum
        t_var = torch.sum(w * t_diff ** 2) / w_sum

        corr = cov / (torch.sqrt(p_var + eps) * torch.sqrt(t_var + eps) + eps)
        target_loss = -corr

        if lambda_anchor > 0.0:
            wmse = torch.sum(w * (p - t) ** 2) / w_sum
            target_loss = target_loss + lambda_anchor * (wmse / (t_var + eps))

        loss = loss + target_loss

    return loss / 2.0


def asym_corr_penalty_loss(pred: torch.Tensor, target: torch.Tensor, lambda_corr: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    base_loss = base_weighted_pearson_loss(pred, target, eps=eps)

    p_flat = pred.reshape(-1, 2)
    t_flat = torch.clamp(target.reshape(-1, 2), -METRIC_CLIP, METRIC_CLIP)

    p0, p1 = p_flat[:, 0] - torch.mean(p_flat[:, 0]), p_flat[:, 1] - torch.mean(p_flat[:, 1])
    t0, t1 = t_flat[:, 0] - torch.mean(t_flat[:, 0]), t_flat[:, 1] - torch.mean(t_flat[:, 1])

    corr_p = torch.sum(p0 * p1) / (torch.sqrt(torch.sum(p0 ** 2) + eps) * torch.sqrt(torch.sum(p1 ** 2) + eps) + eps)
    corr_t = torch.sum(t0 * t1) / (torch.sqrt(torch.sum(t0 ** 2) + eps) * torch.sqrt(torch.sum(t1 ** 2) + eps) + eps)

    corr_penalty = (torch.clamp(corr_p, -1.0, 1.0) - torch.clamp(corr_t, -1.0, 1.0)) ** 2
    return base_loss + lambda_corr * corr_penalty


def local_global_weighted_pearson_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    global_weight: float = 0.7, 
    local_weight: float = 0.3, 
    eps: float = 1e-8
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    t = torch.clamp(target, -METRIC_CLIP, METRIC_CLIP)
    w = torch.abs(t).clamp(min=eps)

    # 1. Global Batch Loss
    p_flat = pred.reshape(-1, 2)
    t_flat = t.reshape(-1, 2)
    w_flat = w.reshape(-1, 2)

    sw_g = torch.sum(w_flat, dim=0, keepdim=True)
    p_mean_g = torch.sum(w_flat * p_flat, dim=0, keepdim=True) / sw_g
    t_mean_g = torch.sum(w_flat * t_flat, dim=0, keepdim=True) / sw_g

    p_diff_g = p_flat - p_mean_g
    t_diff_g = t_flat - t_mean_g

    cov_g = torch.sum(w_flat * p_diff_g * t_diff_g, dim=0, keepdim=True) / sw_g
    p_var_g = torch.sum(w_flat * p_diff_g ** 2, dim=0, keepdim=True) / sw_g
    t_var_g = torch.sum(w_flat * t_diff_g ** 2, dim=0, keepdim=True) / sw_g

    corr_g = cov_g / (torch.sqrt(p_var_g + eps) * torch.sqrt(t_var_g + eps) + eps)
    global_loss = -torch.mean(torch.clamp(corr_g, -1.0, 1.0))

    # 2. Local Sequence Loss
    sw_l = torch.sum(w, dim=1, keepdim=True)
    p_mean_l = torch.sum(w * pred, dim=1, keepdim=True) / sw_l
    t_mean_l = torch.sum(w * t, dim=1, keepdim=True) / sw_l

    p_diff_l = pred - p_mean_l
    t_diff_l = t - t_mean_l

    cov_l = torch.sum(w * p_diff_l * t_diff_l, dim=1, keepdim=True) / sw_l
    p_var_l = torch.sum(w * p_diff_l ** 2, dim=1, keepdim=True) / sw_l
    t_var_l = torch.sum(w * t_diff_l ** 2, dim=1, keepdim=True) / sw_l

    corr_l = cov_l / (torch.sqrt(p_var_l + eps) * torch.sqrt(t_var_l + eps) + eps)
    local_loss = -torch.mean(torch.clamp(corr_l, -1.0, 1.0))

    return global_weight * global_loss + local_weight * local_loss