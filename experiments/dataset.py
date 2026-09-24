import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH

TRAIN_FEAT_PATH = "../datasets/train_features.mmap"
TRAIN_TARG_PATH = "../datasets/train_targets.mmap"


class MmapChunkDataset(Dataset):
    def __init__(self, feat_path=TRAIN_FEAT_PATH, targ_path=TRAIN_TARG_PATH, chunk_size=2000):
        self.feat_path = feat_path
        self.targ_path = targ_path
        self.chunk_size = chunk_size
        self.chunks_per_seq = SEQUENCE_LENGTH // chunk_size
        self.n_feat = len(FEATURE_COLUMNS)
        self.n_targ = len(TARGET_COLUMNS)

        bytes_per_seq = SEQUENCE_LENGTH * self.n_feat * 4
        total_bytes = os.path.getsize(feat_path)
        self.num_seq = total_bytes // bytes_per_seq

        self.features = None
        self.targets = None

    def __len__(self):
        return self.num_seq

    def _init_mmaps(self):
        self.features = np.memmap(
            self.feat_path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_seq, SEQUENCE_LENGTH, self.n_feat)
        )
        self.targets = np.memmap(
            self.targ_path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_seq, SEQUENCE_LENGTH, self.n_targ)
        )

    def __getitem__(self, idx):
        if self.features is None:
            self._init_mmaps()

        valid_len = self.chunks_per_seq * self.chunk_size
        feat_np = self.features[idx, :valid_len]
        targ_np = self.targets[idx, :valid_len]

        feat = torch.from_numpy(feat_np.copy()).view(self.chunks_per_seq, self.chunk_size, self.n_feat)
        targ = torch.from_numpy(targ_np.copy()).view(self.chunks_per_seq, self.chunk_size, self.n_targ)

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

    feat_path = getattr(cfg, "train_feat_mmap", TRAIN_FEAT_PATH)
    targ_path = getattr(cfg, "train_targ_mmap", TRAIN_TARG_PATH)
    ds = MmapChunkDataset(feat_path, targ_path, chunk_size=cfg.chunk_size)
    batch_size = getattr(cfg, "full_batch_size", 4)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_chunk_shuffle,
    )