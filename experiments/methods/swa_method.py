import os
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel
from methods.validator import evaluate


def weighted_pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):

    pred = pred.float()
    target = target.float()

    p_flat = pred.reshape(-1, 2)
    t_flat = torch.clamp(target.reshape(-1, 2), -2.0, 2.0)
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
        loss = loss - corr

    return loss / 2.0


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

    # Инициализация SWA модели
    swa_model = AveragedModel(model)
    swa_start_epoch = getattr(cfg, "swa_start_epoch", 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    num_batches = len(train_loader)
    val_interval = max(1, num_batches // 5)
    val_steps_in_epoch = set([val_interval * k for k in range(1, 5)])
    val_steps_in_epoch.add(num_batches)

    best_sub_wp = -float("inf")
    best_swa_sub_wp = -float("inf")
    patience = 0
    patience_limit = 100
    early_stop_triggered = False

    global_step = 0
    best_checkpoint_path = os.path.join(logger.run_dir, "best_sub_model.pt")
    best_swa_checkpoint_path = os.path.join(logger.run_dir, "best_sub_swa_model.pt")

    logger.info(f"Старт обучения | Батчей в эпохе: {num_batches} | SWA старт с эпохи: {swa_start_epoch}")

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
                loss = weighted_pearson_loss(preds, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step_loss = loss.item()

            # Обновление средних весов SWA
            if epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

            step_loss_acc += step_loss
            step_wp_acc += (-step_loss)
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
                # 1. Валидация текущей модели (1/10)
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

                # 2. Параллельная валидация SWA модели (1/10)
                swa_log_str = ""
                if epoch >= swa_start_epoch:
                    swa_res = evaluate(swa_model, cfg, device, sample_stride=10)
                    swa_wp = swa_res['weighted_pearson']

                    if swa_wp > best_swa_sub_wp:
                        best_swa_sub_wp = swa_wp
                        torch.save(swa_model.state_dict(), best_swa_checkpoint_path)
                        swa_mark = " [NEW BEST SWA!]"
                    else:
                        swa_mark = ""

                    swa_log_str = f" | SWA WP: {swa_wp:.4f} (t0: {swa_res['t0']:.4f}, t1: {swa_res['t1']:.4f}){swa_mark}"
                    logger.log_val_event(epoch, global_step, "sub_10pct_swa", current_lr, swa_res, patience)

                logger.info(
                    f">>> SUB-VAL (1/10) | Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} | "
                    f"LR: {current_lr:.2e} | WP: {sub_wp:.4f} (t0: {sub_res['t0']:.4f}, t1: {sub_res['t1']:.4f}){mark}"
                    f"{swa_log_str}"
                )
                logger.log_val_event(epoch, global_step, "sub_10pct", current_lr, sub_res, patience)

                if patience >= patience_limit:
                    logger.info(f"Early Stopping: достигнут лимит стагнации ({patience_limit} проверок).")
                    early_stop_triggered = True
                    break

                model.train()

        if early_stop_triggered:
            break

    # Финальная оценка на 100% валидации для лучшей базовой модели
    logger.info("Финальная оценка лучшего чекпоинта модели на 100% валидации...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    final_res = evaluate(model, cfg, device, sample_stride=1)
    final_wp = final_res['weighted_pearson']
    logger.info(f"ИТОГ BASE FULL VAL WP: {final_wp:.5f} (t0: {final_res['t0']:.4f}, t1: {final_res['t1']:.4f})")
    logger.log_val_event(epoch, global_step, "final_best_full", current_lr, final_res, patience)

    # Финальная оценка на 100% валидации для лучшей SWA модели
    final_swa_wp = None
    if os.path.exists(best_swa_checkpoint_path):
        logger.info("Финальная оценка лучшего чекпоинта SWA на 100% валидации...")
        swa_model.load_state_dict(torch.load(best_swa_checkpoint_path, map_location=device))
        final_swa_res = evaluate(swa_model, cfg, device, sample_stride=1)
        final_swa_wp = final_swa_res['weighted_pearson']
        logger.info(f"ИТОГ SWA FULL VAL WP:  {final_swa_wp:.5f} (t0: {final_swa_res['t0']:.4f}, t1: {final_swa_res['t1']:.4f})")
        logger.log_val_event(epoch, global_step, "final_best_full_swa", current_lr, final_swa_res, patience)

    summary_data = {
        "seed": seed,
        "best_sub_wp": best_sub_wp,
        "final_full_wp": final_wp,
        "best_swa_sub_wp": best_swa_sub_wp,
        "final_full_swa_wp": final_swa_wp,
        "epochs_trained": epoch,
        "total_steps": global_step,
        "early_stopped": early_stop_triggered
    }
    with open(os.path.join(logger.run_dir, "seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # Возвращаем лучший результат из двух подходов
    return max(final_wp, final_swa_wp) if final_swa_wp is not None else final_wp