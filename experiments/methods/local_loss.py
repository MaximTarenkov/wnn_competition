import os
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from methods.validator import evaluate

METRIC_CLIP = 2.0
WARMUP = 0

def weighted_pearson_loss(y_pred, y_true):
    p_steps = y_pred[:, WARMUP:, :]
    t_steps = torch.clamp(y_true[:, WARMUP:, :], -METRIC_CLIP, METRIC_CLIP)
    eps = 1e-8

    # 1. Global Batch Loss
    pred_flat = p_steps.reshape(-1, 2)
    true_flat = t_steps.reshape(-1, 2)
    global_loss = 0.0
    for i in range(2):
        p = pred_flat[:, i]
        t = true_flat[:, i]
        w = torch.abs(t).clamp(min=eps)
        sw = torch.sum(w)
        sn_p = p - torch.sum(p * w) / sw
        sn_t = t - torch.sum(t * w) / sw
        cov = torch.sum(w * sn_p * sn_t) / sw
        var_p = torch.sum(w * sn_p**2) / sw
        var_t = torch.sum(w * sn_t**2) / sw
        corr = cov / (torch.sqrt(var_p + eps) * torch.sqrt(var_t + eps) + eps)
        global_loss -= torch.clamp(corr, -1.0, 1.0)
    global_loss /= 2.0

    # 2. Local Sequence Loss
    local_loss = 0.0
    for i in range(2):
        p = p_steps[:, :, i]
        t = t_steps[:, :, i]
        w = torch.abs(t).clamp(min=eps)
        sw = torch.sum(w, dim=1, keepdim=True)
        sn_p = p - torch.sum(p * w, dim=1, keepdim=True) / sw
        sn_t = t - torch.sum(t * w, dim=1, keepdim=True) / sw
        cov = torch.sum(w * sn_p * sn_t, dim=1, keepdim=True) / sw
        var_p = torch.sum(w * sn_p**2, dim=1, keepdim=True) / sw
        var_t = torch.sum(w * sn_t**2, dim=1, keepdim=True) / sw
        corr = cov / (torch.sqrt(var_p + eps) * torch.sqrt(var_t + eps) + eps)
        local_loss -= torch.mean(torch.clamp(corr, -1.0, 1.0))
    local_loss /= 2.0

    total_loss = local_loss
    return total_loss, -global_loss.item()


class ExperimentLogger:
    def __init__(self, base_dir, exp_name, seed):
        seed_dir = Path(base_dir) / exp_name / f"seed_{seed}"
        if seed_dir.exists():
            counter = 1
            while (Path(base_dir) / exp_name / f"seed_{seed}_{counter}").exists():
                counter += 1
            seed_dir = Path(base_dir) / exp_name / f"seed_{seed}_{counter}"

        seed_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = str(seed_dir)

        self.logger = logging.getLogger(self.run_dir)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(os.path.join(self.run_dir, "run.log"), encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.csv_path = os.path.join(self.run_dir, "history.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "epoch", "step", "event_type", "lr",
                "train_loss", "train_wp", "val_wp", "t0", "t1", "patience"
            ])

    def info(self, msg):
        self.logger.info(msg)

    def log_train_step(self, epoch, step, lr, train_loss, train_wp):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now_str, epoch, step, "train_step", f"{lr:.6e}",
                f"{train_loss:.6f}", f"{train_wp:.6f}", "", "", "", ""
            ])

    def log_val_event(self, epoch, step, event_type, lr, res, patience):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now_str, epoch, step, event_type, f"{lr:.6e}",
                "", "", f"{res['weighted_pearson']:.6f}",
                f"{res['t0']:.6f}", f"{res['t1']:.6f}", patience
            ])


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
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            preds = out[0] if isinstance(out, tuple) else out

            loss, global_wp = weighted_pearson_loss(preds, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step_loss_acc += loss.item()
            step_wp_acc += (-loss.item())
            step_count += 1

            current_lr = optimizer.param_groups[0]['lr']

            if global_step % cfg.log_interval == 0:
                avg_train_loss = step_loss_acc / step_count
                avg_train_wp = step_wp_acc / step_count
                step_loss_acc = 0.0
                step_wp_acc = 0.0
                step_count = 0

                logger.info(
                    f"Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} (Global {global_step:05d}) | "
                    f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:+.4f} | Train WP: {avg_train_wp:.4f}"
                )
                logger.log_train_step(epoch, global_step, current_lr, avg_train_loss, avg_train_wp)

            if step_in_epoch in val_steps_in_epoch:
                sub_res = evaluate(model, cfg.valid_path, device, sample_stride=10)
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

        if not early_stop_triggered:
            full_res = evaluate(model, cfg.valid_path, device, sample_stride=1)
            full_wp = full_res['weighted_pearson']
            logger.info(
                f"=== FULL VAL (100%) | End of Ep {epoch:02d} | "
                f"WP: {full_wp:.5f} (t0: {full_res['t0']:.4f}, t1: {full_res['t1']:.4f}) ==="
            )
            logger.log_val_event(epoch, global_step, "full_100pct", current_lr, full_res, patience)
            model.train()

        if early_stop_triggered:
            break

    logger.info("Финальная оценка лучшего чекпоинта на 100% валидации...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    final_res = evaluate(model, cfg.valid_path, device, sample_stride=1)
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