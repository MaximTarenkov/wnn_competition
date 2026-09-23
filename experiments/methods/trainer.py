import os
import json
import math
import torch
from torch.optim.swa_utils import AveragedModel
from methods.validator import evaluate
from experiment_logger import ExperimentLogger


def generic_train_seed(
    model,
    train_loader,
    cfg,
    exp_name: str,
    seed: int,
    loss_fn,
    scheduler_type: str | None = None,
    use_swa: bool = False,
    swa_start_epoch: int = 2,
    patience_limit: int = 100,
    loss_step_hook=None,
):
    logger = ExperimentLogger(cfg.runs_dir, exp_name, seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    num_batches = len(train_loader)
    total_steps = cfg.max_epochs * num_batches

    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=getattr(cfg, "min_lr", 1e-6)
        )

    swa_model = AveragedModel(model) if use_swa else None

    val_interval = max(1, num_batches // 5)
    val_steps_in_epoch = set([val_interval * k for k in range(1, 5)])
    val_steps_in_epoch.add(num_batches)

    best_sub_wp = -float("inf")
    best_swa_sub_wp = -float("inf")
    patience = 0
    early_stop_triggered = False

    global_step = 0
    best_checkpoint_path = os.path.join(logger.run_dir, "best_sub_model.pt")
    best_swa_checkpoint_path = os.path.join(logger.run_dir, "best_sub_swa_model.pt")

    logger.info(f"Старт обучения | Батчей: {num_batches} | Всего шагов: {total_steps} | Шедулер: {scheduler_type} | SWA: {use_swa}")

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

            extra_loss_kwargs = {}
            if loss_step_hook is not None:
                extra_loss_kwargs = loss_step_hook(global_step, total_steps, cfg)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                preds = out[0] if isinstance(out, tuple) else out
                loss = loss_fn(preds, y, **extra_loss_kwargs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

            step_loss = loss.item()
            step_loss_acc += step_loss
            step_wp_acc += (-step_loss)
            step_count += 1

            current_lr = optimizer.param_groups[0]["lr"]

            if global_step % cfg.log_interval == 0:
                avg_train_loss = step_loss_acc / step_count
                avg_train_wp = step_wp_acc / step_count
                step_loss_acc, step_wp_acc, step_count = 0.0, 0.0, 0

                extra_str = "".join([f" | {k}: {v:.4f}" for k, v in extra_loss_kwargs.items()])
                logger.info(
                    f"Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} (Global {global_step:05d}) | "
                    f"LR: {current_lr:.2e}{extra_str} | Train Loss: {avg_train_loss:+.4f}"
                )
                logger.log_train_step(epoch, global_step, current_lr, avg_train_loss, avg_train_wp)

            if step_in_epoch in val_steps_in_epoch:
                sub_res = evaluate(model, cfg, device, sample_stride=10)
                sub_wp = sub_res["weighted_pearson"]

                if sub_wp > best_sub_wp:
                    best_sub_wp = sub_wp
                    patience = 0
                    torch.save(model.state_dict(), best_checkpoint_path)
                    mark = " [NEW BEST 1/10!]"
                else:
                    patience += 1
                    mark = f" [No Improvement: {patience}/{patience_limit}]"

                swa_log_str = ""
                if use_swa and epoch >= swa_start_epoch:
                    swa_res = evaluate(swa_model, cfg, device, sample_stride=10)
                    swa_wp = swa_res["weighted_pearson"]
                    if swa_wp > best_swa_sub_wp:
                        best_swa_sub_wp = swa_wp
                        torch.save(swa_model.state_dict(), best_swa_checkpoint_path)
                        swa_mark = " [NEW BEST SWA!]"
                    else:
                        swa_mark = ""
                    swa_log_str = f" | SWA WP: {swa_wp:.4f}{swa_mark}"
                    logger.log_val_event(epoch, global_step, "sub_10pct_swa", current_lr, swa_res, patience)

                logger.info(
                    f">>> SUB-VAL (1/10) | Ep {epoch:02d} | Step {step_in_epoch:04d}/{num_batches} | "
                    f"LR: {current_lr:.2e} | WP: {sub_wp:.4f} (t0: {sub_res['t0']:.4f}, t1: {sub_res['t1']:.4f}){mark}{swa_log_str}"
                )
                logger.log_val_event(epoch, global_step, "sub_10pct", current_lr, sub_res, patience)

                if patience >= patience_limit:
                    logger.info(f"Early Stopping: достигнут лимит стагнации ({patience_limit} проверок).")
                    early_stop_triggered = True
                    break

                model.train()

        if early_stop_triggered:
            break

    logger.info("Финальная оценка лучшего чекпоинта на 100% валидации...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    final_res = evaluate(model, cfg, device, sample_stride=1)
    final_wp = final_res["weighted_pearson"]
    logger.info(f"ИТОГ СИДА FULL VAL WP: {final_wp:.5f} (t0: {final_res['t0']:.4f}, t1: {final_res['t1']:.4f})")
    logger.log_val_event(epoch, global_step, "final_best_full", current_lr, final_res, patience)

    final_swa_wp = None
    if use_swa and os.path.exists(best_swa_checkpoint_path):
        swa_model.load_state_dict(torch.load(best_swa_checkpoint_path, map_location=device))
        final_swa_res = evaluate(swa_model, cfg, device, sample_stride=1)
        final_swa_wp = final_swa_res["weighted_pearson"]
        logger.info(f"ИТОГ SWA FULL VAL WP:  {final_swa_wp:.5f}")

    summary_data = {
        "seed": seed,
        "best_sub_wp": best_sub_wp,
        "final_full_wp": final_wp,
        "best_swa_sub_wp": best_swa_sub_wp if use_swa else None,
        "final_full_swa_wp": final_swa_wp if use_swa else None,
        "epochs_trained": epoch,
        "total_steps": global_step,
        "early_stopped": early_stop_triggered,
    }
    with open(os.path.join(logger.run_dir, "seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    return max(final_wp, final_swa_wp) if final_swa_wp is not None else final_wp