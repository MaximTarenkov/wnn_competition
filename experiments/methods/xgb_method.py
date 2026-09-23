import os
import gc
import json
import numpy as np
import pyarrow.parquet as pq
import torch

from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, WARMUP
from methods.validator import evaluate
from methods.base_method import ExperimentLogger


def _load_train_matrix(cfg, max_seqs: int = 250, step_stride: int = 1, add_deltas: bool = True):
    pf = pq.ParquetFile(cfg.train_path)
    num_seqs = min(max_seqs, pf.num_row_groups) if max_seqs else pf.num_row_groups
    cols = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
    x_list, y_list = [], []

    for idx in range(num_seqs):
        tbl = pf.read_row_group(idx, columns=cols, use_threads=False)
        feat = np.column_stack([tbl[c].to_numpy(zero_copy_only=False) for c in FEATURE_COLUMNS]).astype(np.float32)[WARMUP:]
        targ = np.column_stack([tbl[c].to_numpy(zero_copy_only=False) for c in TARGET_COLUMNS]).astype(np.float32)[WARMUP:]

        if add_deltas:
            delta = np.zeros_like(feat)
            delta[1:] = feat[1:] - feat[:-1]
            feat = np.concatenate([feat, delta], axis=-1)

        # .copy() обязательно, чтобы освободить из RAM исходный массив
        x_list.append(feat[::step_stride].copy())
        y_list.append(targ[::step_stride].copy())

        del tbl, feat, targ
        if (idx + 1) % 50 == 0:
            gc.collect()

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    del x_list, y_list
    gc.collect()
    return x, y


def _load_val_matrix(cfg, sample_stride: int = 10, add_deltas: bool = True):
    bytes_per_seq = SEQUENCE_LENGTH * len(FEATURE_COLUMNS) * 4
    num_seq = os.path.getsize(cfg.val_feat_mmap) // bytes_per_seq
    indices = list(range(0, num_seq, sample_stride))

    feat_mmap = np.memmap(cfg.val_feat_mmap, dtype=np.float32, mode="r", shape=(num_seq, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
    targ_mmap = np.memmap(cfg.val_targ_mmap, dtype=np.float32, mode="r", shape=(num_seq, SEQUENCE_LENGTH, 2))
    mask_mmap = np.memmap(cfg.val_mask_mmap, dtype=bool, mode="r", shape=(num_seq, SEQUENCE_LENGTH))

    x_list, y_list = [], []
    for s_idx in indices:
        feat = np.array(feat_mmap[s_idx])
        targ = np.array(targ_mmap[s_idx])
        mask = np.array(mask_mmap[s_idx])

        if add_deltas:
            delta = np.zeros_like(feat)
            delta[1:] = feat[1:] - feat[:-1]
            feat = np.concatenate([feat, delta], axis=-1)

        x_list.append(feat[mask].copy())
        y_list.append(targ[mask].copy())
        del feat, targ, mask

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    del x_list, y_list
    gc.collect()
    return x, y


def train_seed(model, train_loader, cfg, exp_name: str, seed: int) -> float:
    logger = ExperimentLogger(cfg.runs_dir, exp_name, seed)
    
    max_seqs = getattr(cfg, "xgb_max_seqs", 500)
    stride = getattr(cfg, "xgb_step_stride", 1)

    logger.info(f"Загрузка выборки (seqs={max_seqs}, step_stride={stride})...")
    x_train, y_train = _load_train_matrix(cfg, max_seqs=max_seqs, step_stride=stride, add_deltas=model.add_deltas)
    x_val, y_val = _load_val_matrix(cfg, sample_stride=10, add_deltas=model.add_deltas)

    logger.info(f"Память под X_train: {x_train.nbytes / 1024**2:.1f} MB | Форма: {x_train.shape}")

    y_train_clip = np.clip(y_train, -2.0, 2.0)
    y_val_clip = np.clip(y_val, -2.0, 2.0)
    w_train = np.abs(y_train_clip) + 1e-4

    logger.info("Старт обучения XGBoost...")
    model.fit(x_train, y_train_clip, w_train, x_val, y_val_clip)
    model.save(logger.run_dir)

    del x_train, y_train, w_train, x_val, y_val
    gc.collect()

    logger.info("Финальная валидация на 100% выборке...")
    res = evaluate(model, cfg, cfg.device, sample_stride=1, batch_size=8)
    final_wp = res["weighted_pearson"]

    logger.info(f"ИТОГ XGB FULL VAL WP: {final_wp:.5f} (t0: {res['t0']:.4f}, t1: {res['t1']:.4f})")
    logger.log_val_event(1, 1, "final_best_full", 0.0, res, 0)

    with open(os.path.join(logger.run_dir, "seed_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "final_full_wp": final_wp, "t0": res["t0"], "t1": res["t1"]}, f, indent=4)

    return final_wp