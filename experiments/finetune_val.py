import os
import gc
import random
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH
from models.gru_mlp_encoders_disentangled_l1_delta_silu_hotfix import create_model
from methods.validator import evaluate
from methods.asym_wp_loss import ExperimentLogger  # Используем логгер проекта

# Оптимизации CUDA
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# =====================================================================
# 1. ДАТАСЕТ: 90% ВАЛИДАЦИИ (ИСКЛЮЧАЕМ КАЖДЫЙ 10-Й СЭМПЛ)
# =====================================================================
class ParquetChunkSubsetDataset(Dataset):
    def __init__(self, parquet_path, indices, chunk_size=2000):
        self.parquet_path = parquet_path
        self.indices = indices
        self.chunk_size = chunk_size
        self.chunks_per_seq = SEQUENCE_LENGTH // chunk_size
        self.n_feat = len(FEATURE_COLUMNS)
        self.n_targ = len(TARGET_COLUMNS)
        self.columns_to_load = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
        self.parquet = pq.ParquetFile(self.parquet_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_group_idx = self.indices[idx]
        table = self.parquet.read_row_group(
            real_group_idx, columns=self.columns_to_load, use_threads=False
        )

        feat_np = np.empty((SEQUENCE_LENGTH, self.n_feat), dtype=np.float32)
        for i, col in enumerate(FEATURE_COLUMNS):
            feat_np[:, i] = table[col].to_numpy(zero_copy_only=False)

        targ_np = np.empty((SEQUENCE_LENGTH, self.n_targ), dtype=np.float32)
        for i, col in enumerate(TARGET_COLUMNS):
            targ_np[:, i] = table[col].to_numpy(zero_copy_only=False)

        valid_len = self.chunks_per_seq * self.chunk_size
        feat = torch.from_numpy(feat_np[:valid_len]).view(
            self.chunks_per_seq, self.chunk_size, self.n_feat
        )
        targ = torch.from_numpy(targ_np[:valid_len]).view(
            self.chunks_per_seq, self.chunk_size, self.n_targ
        )

        return feat, targ


def collate_chunk_shuffle(batch):
    features = torch.cat([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)
    perm = torch.randperm(features.size(0))
    return features[perm], targets[perm]


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# 2. ОСНОВНОЙ ПАЙПЛАЙН ДООБУЧЕНИЯ
# =====================================================================
def finetune():
    cfg = Config()
    SEED = 28775
    set_seed(SEED)

    CHECKPOINT_PATH = "runs/gru_mlp_encoders_disentangled_l1_delta_silu_hotfix_coslr_5b/seed_28775/best_sub_model.pt"
    EXP_NAME = "finetune_hotfix_val_1ep"

    # Параметры файнтюнинга
    FT_START_LR = 5e-5      # Мягкий стартовый LR для дообучения
    FT_MIN_LR = 1e-6        # Финальный LR к концу эпохи
    WEIGHT_DECAY = 1e-4

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger = ExperimentLogger(cfg.runs_dir, EXP_NAME, SEED)

    # 1. Собираем индексы 90% выборки (вырезаем stride 10)
    parquet_val = pq.ParquetFile(cfg.valid_path)
    total_val_seqs = parquet_val.num_row_groups
    train_val_indices = [i for i in range(total_val_seqs) if i % 10 != 0]

    logger.info(f"Всего последовательностей в Valid: {total_val_seqs}")
    logger.info(f"Используем для дообучения (90%): {len(train_val_indices)}")
    logger.info(f"Отложено для контроля метрики (10%): {total_val_seqs - len(train_val_indices)}")

    ft_dataset = ParquetChunkSubsetDataset(
        cfg.valid_path, indices=train_val_indices, chunk_size=cfg.chunk_size
    )
    ft_loader = DataLoader(
        ft_dataset,
        batch_size=cfg.full_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_chunk_shuffle,
    )

    # 2. Загружаем модель
    logger.info(f"Загрузка базового чекпоинта: {CHECKPOINT_PATH}")
    model = create_model(cfg).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # Начальная проверка на отложенном саб-сете до обучения
    init_res = evaluate(model, cfg, device, sample_stride=10)
    logger.info(f"Начальный WP на контрольном 10% сете: {init_res['weighted_pearson']:.5f}")

    # 3. Оптимизатор и Шедулер на 1 эпоху
    optimizer = torch.optim.AdamW(model.parameters(), lr=FT_START_LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(ft_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=FT_MIN_LR
    )

    # 5 валидаций за эпоху
    val_interval = max(1, total_steps // 5)
    val_steps = set([val_interval * k for k in range(1, 5)])
    val_steps.add(total_steps)

    best_sub_wp = init_res['weighted_pearson']
    best_checkpoint_path = os.path.join(logger.run_dir, "best_finetuned_model.pt")

    logger.info(f"Старт 1 эпохи дообучения | Всего шагов: {total_steps} | Валидации на шагах: {sorted(list(val_steps))}")

    model.train()
    step_loss_acc = 0.0
    step_count = 0

    for step, (x, y) in enumerate(ft_loader, 1):
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
        scheduler.step()

        step_loss_acc += loss.item()
        step_count += 1
        current_lr = optimizer.param_groups[0]['lr']

        if step % cfg.log_interval == 0:
            avg_loss = step_loss_acc / step_count
            logger.info(f"Step {step:04d}/{total_steps} | LR: {current_lr:.2e} | Train WP: {-avg_loss:.4f}")
            step_loss_acc = 0.0
            step_count = 0

        # Валидация 5 раз за эпоху на нетронутом 10% сабсете
        if step in val_steps:
            sub_res = evaluate(model, cfg, device, sample_stride=10)
            sub_wp = sub_res['weighted_pearson']

            if sub_wp > best_sub_wp:
                best_sub_wp = sub_wp
                torch.save(model.state_dict(), best_checkpoint_path)
                mark = " [🔥 NEW BEST SUB-VAL!]"
            else:
                mark = ""

            logger.info(
                f">>> CHECKPOINT VAL (1/10) | Step {step:04d}/{total_steps} | "
                f"LR: {current_lr:.2e} | WP: {sub_wp:.5f} (t0: {sub_res['t0']:.4f}, t1: {sub_res['t1']:.4f}){mark}"
            )
            model.train()

    # Сохраняем финальную версию после конца эпохи
    final_epoch_checkpoint_path = os.path.join(logger.run_dir, "last_step_finetuned_model.pt")
    torch.save(model.state_dict(), final_epoch_checkpoint_path)

    logger.info("=" * 60)
    logger.info(f"Дообучение завершено!")
    logger.info(f"Лучший чекпоинт по 10% Val сохранен в: {best_checkpoint_path} (WP: {best_sub_wp:.5f})")
    logger.info(f"Финальный чекпоинт сохранен в: {final_epoch_checkpoint_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    finetune()