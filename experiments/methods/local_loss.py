import os
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from methods.validator import evaluate
from experiment_logger import ExperimentLogger


def weighted_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):

    pred = pred.float()
    target = target.float()

    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    t = torch.clamp(target, -2.0, 2.0)
    w = torch.abs(t).clamp(min=eps)

    sw = torch.sum(w, dim=1, keepdim=True)
    p_mean = torch.sum(w * pred, dim=1, keepdim=True) / sw
    t_mean = torch.sum(w * t, dim=1, keepdim=True) / sw

    p_diff = pred - p_mean
    t_diff = t - t_mean

    cov = torch.sum(w * p_diff * t_diff, dim=1, keepdim=True) / sw
    p_var = torch.sum(w * p_diff ** 2, dim=1, keepdim=True) / sw
    t_var = torch.sum(w * t_diff ** 2, dim=1, keepdim=True) / sw

    corr = cov / (torch.sqrt(p_var + eps) * torch.sqrt(t_var + eps) + eps)
    corr = torch.clamp(corr, -1.0, 1.0)
    return -torch.mean(corr)





def train_seed(model, train_loader, cfg, exp_name, seed):
    logger = ExperimentLogger(cfg.runs_dir, exp_name, seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    num_batches = len(train_loader)
    val_interval = max(1, num_batches // 5)
    val_steps_in_epoch = set([val_interval * k for k in range(1, 5)])
    val_steps_in_epoch.add(num_batches)

    best_sub_wp = -float("inf")
    patience = 0
    patience_limit = 10
    early_stop_triggered = False

    global_step = 0
    best_checkpoint_path = os.path.join(logger.run_dir, "best_sub_model.pt")

    logger.info(f"Старт обучения | Батчей в эпохе: {num_batches}")

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

            if x.dim() == 4:
                B, num_chunks, T_chunk, D = x.shape
                h = None
                preds_list = []

                for c in range(num_chunks):
                    x_chunk = x[:, c, :, :]

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        out = model(x_chunk, h)
                        if isinstance(out, tuple):
                            pred_chunk, h = out
                        else:
                            pred_chunk, h = out, None

                    preds_list.append(pred_chunk)

                    if h is not None:
                        h = h.detach() 

                all_preds = torch.cat(preds_list, dim=1)
                y_full = y.view(B, -1, y.shape[-1])

                loss = weighted_pearson_loss(all_preds, y_full)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                step_loss = loss.detach()

            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x)
                    preds = out[0] if isinstance(out, tuple) else out

                loss = weighted_pearson_loss(preds, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                step_loss = loss.detach()

            step_loss_acc += step_loss
            step_wp_acc += (-step_loss)
            step_count += 1

            current_lr = optimizer.param_groups[0]['lr']

            if global_step % cfg.log_interval == 0:
                avg_train_loss = (step_loss_acc / step_count).item()
                avg_train_wp = -avg_train_loss
                step_loss_acc = 0.0
                step_wp_acc = 0.0
                step_count = 0

                logger.info(
                    f"Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} (Global {global_step:05d}) | "
                    f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:+.4f} | Train WP: {avg_train_wp:.4f}"
                )
                logger.log_train_step(epoch, global_step, current_lr, avg_train_loss, avg_train_wp)

            if step_in_epoch in val_steps_in_epoch:
                sub_res = evaluate(model, cfg, device, sample_stride=10)
                sub_wp = sub_res['weighted_pearson']

                if sub_wp > best_sub_wp:
                    best_sub_wp = sub_wp
                    patience = 0
                    torch.save(model.state_dict(), best_checkpoint_path)
                    mark = " [NEW BEST 1/10!]"
                else:
                    patience += 1
                    mark = f" [No Improvement: {patience}/{patience_limit}]"

                logger.info(
                    f">>> SUB-VAL (1/10) | Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} | "
                    f"LR: {current_lr:.2e} | WP: {sub_wp:.4f} (t0: {sub_res['t0']:.4f}, t1: {sub_res['t1']:.4f}){mark}"
                )
                logger.log_val_event(epoch, global_step, "sub_10pct", current_lr, sub_res, patience)

                if patience >= patience_limit:
                    logger.info(f"Early Stopping: достигнут лимит стагнации ({patience_limit} проверок).")
                    early_stop_triggered = True
                    break

                model.train()

        # if not early_stop_triggered:
        #     full_res = evaluate(model, cfg, device, sample_stride=1)
        #     full_wp = full_res['weighted_pearson']
        #     logger.info(
        #         f"=== FULL VAL (100%) | End of Ep {epoch:02d} | "
        #         f"WP: {full_wp:.5f} (t0: {full_res['t0']:.4f}, t1: {full_res['t1']:.4f}) ==="
        #     )
        #     logger.log_val_event(epoch, global_step, "full_100pct", current_lr, full_res, patience)
        #     model.train()

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
        "early_stopped": early_stop_triggered
    }
    with open(os.path.join(logger.run_dir, "seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    return final_wp