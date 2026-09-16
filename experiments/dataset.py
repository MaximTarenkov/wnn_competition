import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH


class MemmapFastChunkDataset(Dataset):
    def __init__(self, feat_path, targ_path, chunk_size=2000):
        self.feat_path = feat_path
        self.targ_path = targ_path
        self.chunk_size = chunk_size
        self.chunks_per_seq = SEQUENCE_LENGTH // chunk_size
        self.n_feat = len(FEATURE_COLUMNS)
        self.n_targ = len(TARGET_COLUMNS)

        feat_bytes = os.path.getsize(feat_path)
        bytes_per_seq = self.chunks_per_seq * self.chunk_size * self.n_feat * 4 
        self.num_seq = feat_bytes // bytes_per_seq

        self.features = None
        self.targets = None

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
        feat = torch.from_numpy(self.features[idx].copy())
        targ = torch.from_numpy(self.targets[idx].copy())
        return feat, targ


def chunk_collate_fn(batch, shuffle_chunks=False):
    features = torch.cat([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)

    if shuffle_chunks:
        perm = torch.randperm(features.size(0))
        return features[perm], targets[perm]

    return features, targets


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_train_dataloader(cfg, seed=None, shuffle_chunks=False) -> DataLoader:
    ds = MemmapFastChunkDataset(
        feat_path=cfg.train_feat_mmap,
        targ_path=cfg.train_targ_mmap,
        chunk_size=cfg.chunk_size
    )
    
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    collate = lambda b: chunk_collate_fn(b, shuffle_chunks=shuffle_chunks)

    return DataLoader(
        ds,
        batch_size=getattr(cfg, "full_batch_size", 4),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate,
    )