import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH


class MemmapChunkDataset(Dataset):
    def __init__(self, feat_path, targ_path, chunk_size=2000):
        self.feat_path = feat_path
        self.targ_path = targ_path
        self.chunk_size = chunk_size
        self.chunks_per_seq = SEQUENCE_LENGTH // chunk_size
        self.n_feat = len(FEATURE_COLUMNS)
        self.n_targ = len(TARGET_COLUMNS)

        feat_bytes = os.path.getsize(feat_path)
        bytes_per_seq = SEQUENCE_LENGTH * self.n_feat * 4 
        self.num_seq = feat_bytes // bytes_per_seq

        self.features: np.memmap | None = None
        self.targets: np.memmap | None = None

    def _init_mmap(self):
        if self.features is None:
            self.features = np.memmap(
                self.feat_path, dtype=np.float32, mode="r",
                shape=(self.num_seq, self.chunks_per_seq, self.chunk_size, self.n_feat)
            )
            self.targets = np.memmap(
                self.targ_path, dtype=np.float32, mode="r",
                shape=(self.num_seq, self.chunks_per_seq, self.chunk_size, self.n_targ)
            )

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        self._init_mmap()

        assert self.features is not None
        assert self.targets is not None

        feat = torch.from_numpy(self.features[idx].copy())
        targ = torch.from_numpy(self.targets[idx].copy())
        return feat, targ


class MemmapFullDataset(Dataset):
    def __init__(self, feat_path, targ_path):
        self.feat_path = feat_path
        self.targ_path = targ_path
        self.n_feat = len(FEATURE_COLUMNS)
        self.n_targ = len(TARGET_COLUMNS)

        feat_bytes = os.path.getsize(feat_path)
        bytes_per_seq = SEQUENCE_LENGTH * self.n_feat * 4 
        self.num_seq = feat_bytes // bytes_per_seq

        self.features: np.memmap | None = None
        self.targets: np.memmap | None = None

    def _init_mmap(self):
        if self.features is None:
            self.features = np.memmap(
                self.feat_path, dtype=np.float32, mode="r",
                shape=(self.num_seq, SEQUENCE_LENGTH, self.n_feat)
            )
            self.targets = np.memmap(
                self.targ_path, dtype=np.float32, mode="r",
                shape=(self.num_seq, SEQUENCE_LENGTH, self.n_targ)
            )

    def __len__(self):
        return self.num_seq

    def __getitem__(self, idx):
        self._init_mmap()

        assert self.features is not None
        assert self.targets is not None

        feat = torch.from_numpy(self.features[idx].copy())
        targ = torch.from_numpy(self.targets[idx].copy())
        return feat, targ



def collate_chunk_shuffle(batch):
    features = torch.cat([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)
    perm = torch.randperm(features.size(0))
    return features[perm], targets[perm]


def collate_chunk_noshuffle(batch):
    features = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return features, targets


def collate_full(batch):
    features = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return features, targets



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_dataloader(cfg, mode: str = "chunk_shuffle", seed: int | None = None) -> DataLoader:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    if mode == "chunk_shuffle":
        ds = MemmapChunkDataset(cfg.train_feat_mmap, cfg.train_targ_mmap, chunk_size=cfg.chunk_size)
        collate_fn = collate_chunk_shuffle
        batch_size = getattr(cfg, "full_batch_size", 4)

    elif mode == "chunk_noshuffle":
        ds = MemmapChunkDataset(cfg.train_feat_mmap, cfg.train_targ_mmap, chunk_size=cfg.chunk_size)
        collate_fn = collate_chunk_noshuffle
        batch_size = getattr(cfg, "full_batch_size", 4)

    elif mode == "full":
        ds = MemmapFullDataset(cfg.train_feat_mmap, cfg.train_targ_mmap)
        collate_fn = collate_full
        batch_size = getattr(cfg, "full_seq_batch_size", 2)

    else:
        raise ValueError(f"Неизвестный режим датасета: {mode}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn,
    )