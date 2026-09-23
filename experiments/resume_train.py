import os
import gc
import json
import math
import random
from pathlib import Path
import numpy as np
import torch

from config import Config
from models.gru_mlp_encoders_disentangled_l1_delta_silu_hotfix import create_model
from methods.validator import evaluate
from methods.cosine_scheduler_method import weighted_pearson_loss, ExperimentLogger
import dataset

# Оптимизации CUDA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_warmup_cosine_lr(step, total_steps, warmup_steps=150, max_lr=3.0e-5, min_lr=1.0e-6):
    """Мягкий Linear Warmup, затем Cosine Decay до min_lr."""
    if step <= warmup_steps:
        alpha = step / max(1, warmup_steps)
        return min_lr + (max_lr - min_lr) * alpha
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (max_lr - min_lr) * cosine_decay


def main():
    cfg = Config()
    SEED = 28775
    set_seed(SEED)

    # 1. Параметры эксперимента
    BASE_EXP = "gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_5b"
    CKPT_PATH = f"runs/{BASE_EXP}/seed_{SEED}/best_sub_model.pt"
    RESUME_EXP_NAME = f"clean_resume_hotfix_seed_{SEED}_ep6_7"

    ADDITIONAL_EPOCHS = 2   # 6-я и 7-я эпохи
    WARMUP_STEPS = 150      # Шагов мягкого разгона оптимизатора
    MAX_LR = 3.0e-5         # Пиковый LR
    MIN_LR = 1.0e-6         # Финальный LR

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(cfg.runs_dir, RESUME_EXP_NAME, SEED)

    logger.info("=" * 75)
    logger.info(f"СТАРТ RESUME ОБУЧЕНИЯ (СИД {SEED})")
    logger.info(f"Базовый чекпоинт: {CKPT_PATH}")
    logger.info("=" * 75)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Файл {CKPT_PATH} не найден! Проверьте путь.")

    # 2. Загрузка исходной модели
    model = create_model(cfg).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    # 3. Фиксация стартового бейзлайна
    logger.info("Замер базовой метрики загруженного чекпоинта на 1/10 валидации...")
    init_res = evaluate(model, cfg, device, sample_stride=10)
    best_sub_wp = init_res["weighted_pearson"]
    logger.info(f" Стартовый базовый рекорд: WP = {best_sub_wp:.5f} (t0: {init_res['t0']:.4f}, t1: {init_res['t1']:.4f})")

    # 4. Даталоадер со сдвинутым сидом
    DATA_SEED = SEED + 500
    train_loader = dataset.get_train_dataloader(cfg, mode="chunk_shuffle", seed=DATA_SEED)
    num_batches = len(train_loader)
    total_steps = ADDITIONAL_EPOCHS * num_batches

    # 5. Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=MIN_LR, weight_decay=cfg.weight_decay)

    val_interval = max(1, num_batches // 5)
    val_steps_in_epoch = set([val_interval * k for k in range(1, 5)])
    val_steps_in_epoch.add(num_batches)

    global_step = 0
    best_checkpoint_path = os.path.join(logger.run_dir, "best_resumed_model.pt")
    last_checkpoint_path = os.path.join(logger.run_dir, "last_model.pt")
    improved = False

    logger.info(f"Всего батчей в эпохе: {num_batches} | Всего шагов: {total_steps}")
    logger.info(f"План LR: {MIN_LR:.1e} --(warmup {WARMUP_STEPS} st)--> {MAX_LR:.1e} --(cosine)--> {MIN_LR:.1e}")

    step_loss_acc = 0.0
    step_wp_acc = 0.0
    step_count = 0

    for epoch_idx in range(1, ADDITIONAL_EPOCHS + 1):
        actual_epoch = 5 + epoch_idx  # 6 и 7 эпохи
        model.train()

        for step_in_epoch, (x, y) in enumerate(train_loader, 1):
            global_step += 1

            current_lr = compute_warmup_cosine_lr(
                step=global_step,
                total_steps=total_steps,
                warmup_steps=WARMUP_STEPS,
                max_lr=MAX_LR,
                min_lr=MIN_LR,
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

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
            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x)
                    preds = out[0] if isinstance(out, tuple) else out

                loss = weighted_pearson_loss(preds, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            step_loss = loss.item()
            step_loss_acc += step_loss
            step_wp_acc += (-step_loss)
            step_count += 1

            if global_step % cfg.log_interval == 0:
                avg_train_loss = step_loss_acc / step_count
                avg_train_wp = step_wp_acc / step_count
                step_loss_acc = 0.0
                step_wp_acc = 0.0
                step_count = 0

                logger.info(
                    f"Ep {actual_epoch:02d} | Step {step_in_epoch:04d}/{num_batches} (Global {global_step:04d}/{total_steps}) | "
                    f"LR: {current_lr:.2e} | Train WP: {avg_train_wp:.4f}"
                )
                logger.log_train_step(actual_epoch, global_step, current_lr, avg_train_loss, avg_train_wp)

            # Валидация 5 раз за эпоху
            if step_in_epoch in val_steps_in_epoch:
                sub_res = evaluate(model, cfg, device, sample_stride=10)
                sub_wp = sub_res['weighted_pearson']

                if sub_wp > best_sub_wp:
                    diff = sub_wp - best_sub_wp
                    best_sub_wp = sub_wp
                    improved = True
                    torch.save(model.state_dict(), best_checkpoint_path)
                    mark = f" [🔥 НОВЫЙ РЕКОРД! +{diff:.5f}]"
                else:
                    mark = f" [Старый рекорд: {best_sub_wp:.5f}]"

                logger.info(
                    f">>> SUB-VAL (1/10) | Ep {actual_epoch:02d} | Step {step_in_epoch:04d}/{num_batches} | "
                    f"LR: {current_lr:.2e} | WP: {sub_wp:.5f} (t0: {sub_res['t0']:.4f}, t1: {sub_res['t1']:.4f}){mark}"
                )
                logger.log_val_event(actual_epoch, global_step, "sub_10pct", current_lr, sub_res, 0)
                model.train()

    # =========================================================================
    # 6. СОХРАНЕНИЕ И ПОЛНАЯ ОЦЕНКА (100% ВАЛИДАЦИЯ ДЛЯ ОБЕИХ МОДЕЛЕЙ)
    # =========================================================================
    logger.info("=" * 75)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО. НАЧИНАЕТСЯ ФИНАЛЬНАЯ ВЕРИФИКАЦИЯ НА 100% ВАЛИДАЦИИ.")
    logger.info("=" * 75)

    # 1. Сохраняем модель с ПОСЛЕДНЕЙ итерации
    torch.save(model.state_dict(), last_checkpoint_path)
    logger.info(f" Чекпоинт последней итерации сохранен: {last_checkpoint_path}")

    # 2. Оценка LAST модели на 100% валидации
    logger.info("\n>>> [1/2] Запуск оценки LAST модели на 100% валидации (sample_stride=1)...")
    last_res = evaluate(model, cfg, device, sample_stride=1)
    last_full_wp = last_res['weighted_pearson']
    logger.info(f"ИТОГ LAST FULL VAL WP:  {last_full_wp:.5f} (t0: {last_res['t0']:.4f}, t1: {last_res['t1']:.4f})")
    logger.log_val_event(actual_epoch, global_step, "final_last_full", current_lr, last_res, 0)

    # 3. Оценка BEST модели на 100% валидации (если она была зафиксирована)
    best_full_wp = None
    if improved and os.path.exists(best_checkpoint_path):
        logger.info("\n>>> [2/2] Запуск оценки BEST модели на 100% валидации (sample_stride=1)...")
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        best_res = evaluate(model, cfg, device, sample_stride=1)
        best_full_wp = best_res['weighted_pearson']
        logger.info(f"ИТОГ BEST FULL VAL WP:  {best_full_wp:.5f} (t0: {best_res['t0']:.4f}, t1: {best_res['t1']:.4f})")
        logger.log_val_event(actual_epoch, global_step, "final_best_full", current_lr, best_res, 0)
    else:
        logger.info("\n>>> Модель не побила рекорд на саб-вале, поэтому отдельный BEST чекпоинт не создавался.")

    # 4. Итоговая сводка и выбор победителя
    logger.info("=" * 75)
    logger.info("ИТОГОВОЕ СРАВНЕНИЕ НА 100% ВАЛИДАЦИИ:")
    logger.info(f"1. LAST MODEL (последний шаг LR 1e-6):  WP = {last_full_wp:.5f}")
    if best_full_wp is not None:
        logger.info(f"2. BEST MODEL (лучший шаг по 10% валу): WP = {best_full_wp:.5f}")

        if last_full_wp >= best_full_wp:
            winner = ("LAST MODEL", last_checkpoint_path, last_full_wp)
        else:
            winner = ("BEST MODEL", best_checkpoint_path, best_full_wp)
        logger.info(f"🏆 ПОБЕДИТЕЛЬ: {winner[0]} с результатом WP = {winner[2]:.5f}")
        logger.info(f"Файл для сабмита: {winner[1]}")
    else:
        logger.info(f"🏆 Готовая модель: {last_checkpoint_path} (WP = {last_full_wp:.5f})")

    logger.info("=" * 75)

    summary_data = {
        "seed": SEED,
        "best_sub_wp": best_sub_wp,
        "last_full_wp": last_full_wp,
        "best_full_wp": best_full_wp,
        "improved_on_subval": improved,
        "epochs_trained": actual_epoch,
        "total_steps": global_step,
    }
    with open(os.path.join(logger.run_dir, "resume_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)


if __name__ == "__main__":
    main()
