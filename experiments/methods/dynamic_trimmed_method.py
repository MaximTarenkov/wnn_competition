import os
import json
import torch
import torch.nn as nn
from methods.validator import evaluate
from experiment_logger import ExperimentLogger


def dynamic_trimmed_weighted_pearson_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    keep_ratio: float = 0.70,
    soft_weight: float = 0.05,
    eps: float = 1e-8
):
    """
    Динамический 70/30 Trimmed Weighted Pearson Loss.
    70% наименее ошибочных точек батча получают 100% веса.
    30% худших выбросов демпфируются коэффициентом soft_weight (0.05).
    """
    pred = pred.float()
    target = target.float()

    p_flat = pred.reshape(-1, 2)
    t_flat = torch.clamp(target.reshape(-1, 2), -2.0, 2.0)
    w_raw = torch.abs(t_flat).clamp(min=eps)

    # Точечная квадратичная невязка с весом
    point_err = w_raw * (p_flat - t_flat) ** 2  # (N, 2)

    loss = 0.0
    for i in range(2):
        err_i = point_err[:, i]
        n_points = err_i.size(0)
        k = int(n_points * keep_ratio)

        # Порог 70-го процентиля ошибки (detach, чтобы не дифференцировать через ранг)
        threshold = torch.kthvalue(err_i, k).values.detach()

        # Маска 70% лучших точек
        mask_easy = (err_i <= threshold).float()

        # Эффективный вес: 1.0 для 70% легких и 0.05 для 30% сложных выбросов
        w = w_raw[:, i] * (mask_easy + soft_weight * (1.0 - mask_easy))

        w_sum = torch.sum(w)
        p_mean = torch.sum(w * p_flat[:, i]) / w_sum
        t_mean = torch.sum(w * t_flat[:, i]) / w_sum
        p_diff = p_flat[:, i] - p_mean
        t_diff = t_flat[:, i] - t_mean

        cov = torch.sum(w * p_diff * t_diff) / w_sum
        p_var = torch.sum(w * p_diff ** 2) / w_sum
        t_var = torch.sum(w * t_diff ** 2) / w_sum

        corr = cov / (torch.sqrt(p_var + eps) * torch.sqrt(t_var + eps) + eps)
        loss = loss - corr

    return loss / 2.0


def train_seed(model, train_loader, cfg, exp_name, seed):
    logger = ExperimentLogger(cfg.runs_dir, exp_name, seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    num_batches = len(train_loader)
    total_steps = cfg.max_epochs * num_batches

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=getattr(cfg, "min_lr", 1e-6)
    )

    val_interval = max(1, num_batches // 5)
    val_steps_in_epoch = set([val_interval * k for k in range(1, 5)])
    val_steps_in_epoch.add(num_batches)

    best_sub_wp = -float("inf")
    patience = 0
    patience_limit = 100
    early_stop_triggered = False

    global_step = 0
    best_checkpoint_path = os.path.join(logger.run_dir, "best_sub_model.pt")

    logger.info(f"Старт Dynamic Trimmed 70/30 | Батчей: {num_batches} | Шагов: {total_steps}")

    step_loss_acc = 0.0
    step_wp_acc = 0.0
    step_count = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()

        for step_in_epoch, (x, y) in enumerate(train_loader, 1):
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(x)
                preds = out[0] if isinstance(out, tuple) else out
                loss = dynamic_trimmed_weighted_pearson_loss(preds, y, keep_ratio=0.70, soft_weight=0.05)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step_loss = loss.item()
            step_loss_acc += step_loss
            step_wp_acc += (-step_loss)
            step_count += 1

            current_lr = optimizer.param_groups[0]['lr']

            if global_step % cfg.log_interval == 0:
                avg_train_loss = step_loss_acc / step_count
                avg_train_wp = step_wp_acc / step_count
                step_loss_acc, step_wp_acc, step_count = 0.0, 0.0, 0
                logger.info(
                    f"Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} | "
                    f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:+.4f} | WP: {avg_train_wp:.4f}"
                )
                logger.log_train_step(epoch, global_step, current_lr, avg_train_loss, avg_train_wp)

            if step_in_epoch in val_steps_in_epoch:
                sub_res = evaluate(model, cfg, device, sample_stride=10)
                sub_wp = sub_res['weighted_pearson']

                if sub_wp > best_sub_wp:
                    best_sub_wp = sub_wp
                    patience = 0
                    torch.save(model.state_dict(), best_checkpoint_path)
                    mark = " [NEW BEST!]"
                else:
                    patience += 1
                    mark = f" [{patience}/{patience_limit}]"

                logger.info(
                    f">>> SUB-VAL | Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} | "
                    f"LR: {current_lr:.2e} | WP: {sub_wp:.4f} (t0: {sub_res['t0']:.4f}, t1: {sub_res['t1']:.4f}){mark}"
                )
                logger.log_val_event(epoch, global_step, "sub_10pct", current_lr, sub_res, patience)

                if patience >= patience_limit:
                    logger.info("Early Stopping triggered.")
                    early_stop_triggered = True
                    break

                model.train()

        if early_stop_triggered:
            break

    logger.info("Финальная оценка лучшего чекпоинта на 100% валидации...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    final_res = evaluate(model, cfg, device, sample_stride=1)
    final_wp = final_res['weighted_pearson']

    logger.info(f"ИТОГ СИДА FULL VAL WP: {final_wp:.5f} (t0: {final_res['t0']:.4f}, t1: {final_res['t1']:.4f})")
    logger.log_val_event(epoch, global_step, "final_best_full", current_lr, final_res, patience)

    summary_data = {
        "seed": seed,
        "best_sub_wp": best_sub_wp,
        "final_full_wp": final_wp,
        "epochs_trained": epoch,
        "total_steps": global_step,
    }
    with open(os.path.join(logger.run_dir, "seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    return final_wp