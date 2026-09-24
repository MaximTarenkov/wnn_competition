import os
import random
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH

TRAIN_PARQUET_PATH = "../datasets/train.parquet"
VALID_PARQUET_PATH = "../datasets/valid.parquet"

USE_ALIVE_MASK = False
ALIVE_MASK_PATH = os.path.join(os.path.dirname(__file__), "analysis/alive_chunks_mask.npy")
ALIVE_MASK = np.load(ALIVE_MASK_PATH) if (USE_ALIVE_MASK and os.path.exists(ALIVE_MASK_PATH)) else None


class ParquetChunkDataset(Dataset):
    def __init__(self, parquet_path=TRAIN_PARQUET_PATH, chunk_size=2000):
        self.parquet_path = parquet_path
        self.chunk_size = chunk_size
        self.chunks_per_seq = SEQUENCE_LENGTH // chunk_size
        self.n_feat = len(FEATURE_COLUMNS)
        self.n_targ = len(TARGET_COLUMNS)
        self.columns_to_load = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)

        self.parquet = pq.ParquetFile(self.parquet_path)
        self.num_seq = self.parquet.num_row_groups

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        table = self.parquet.read_row_group(idx, columns=self.columns_to_load, use_threads=False)

        feat_np = np.empty((SEQUENCE_LENGTH, self.n_feat), dtype=np.float32)
        for i, col in enumerate(FEATURE_COLUMNS):
            feat_np[:, i] = table[col].to_numpy(zero_copy_only=False)

        targ_np = np.empty((SEQUENCE_LENGTH, self.n_targ), dtype=np.float32)
        for i, col in enumerate(TARGET_COLUMNS):
            targ_np[:, i] = table[col].to_numpy(zero_copy_only=False)

        valid_len = self.chunks_per_seq * self.chunk_size
        feat = torch.from_numpy(feat_np[:valid_len]).view(self.chunks_per_seq, self.chunk_size, self.n_feat)
        targ = torch.from_numpy(targ_np[:valid_len]).view(self.chunks_per_seq, self.chunk_size, self.n_targ)

        if ALIVE_MASK is not None:
            mask = ALIVE_MASK[idx * self.chunks_per_seq : (idx + 1) * self.chunks_per_seq]
            feat, targ = feat[mask], targ[mask]

        return feat, targ


def collate_chunk_shuffle(batch):
    features = torch.cat([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)
    perm = torch.randperm(features.size(0))
    return features[perm], targets[perm]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_dataloader(cfg, seed: int | None = None) -> DataLoader:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    train_path = getattr(cfg, "train_path", TRAIN_PARQUET_PATH)
    ds = ParquetChunkDataset(train_path, chunk_size=cfg.chunk_size)
    batch_size = getattr(cfg, "full_batch_size", 4)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=g,
        collate_fn=collate_chunk_shuffle,
    )