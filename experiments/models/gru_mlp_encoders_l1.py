import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS


class ParquetFastChunkDataset(Dataset):
    def __init__(self, parquet_path, chunk_size=2000):
        self.parquet_path = parquet_path
        parquet_file = pq.ParquetFile(parquet_path)
        self.num_sequences = parquet_file.num_row_groups
        self.chunk_size = chunk_size
        self.chunks_per_seq = SEQUENCE_LENGTH // chunk_size
        self.parquet = None
        self.load_columns = FEATURE_COLUMNS + list(TARGET_COLUMNS)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.parquet is None:
            self.parquet = pq.ParquetFile(self.parquet_path)

        table = self.parquet.read_row_group(idx, columns=self.load_columns)
        features = np.column_stack(
            [table[c].to_numpy() for c in FEATURE_COLUMNS]
        ).astype(np.float32)
        targets = np.column_stack(
            [table[c].to_numpy() for c in TARGET_COLUMNS]
        ).astype(np.float32)

        feat_tensor = torch.from_numpy(features).view(
            self.chunks_per_seq, self.chunk_size, -1
        )
        targ_tensor = torch.from_numpy(targets).view(
            self.chunks_per_seq, self.chunk_size, -1
        )

        return feat_tensor, targ_tensor


def chunk_collate_fn(batch):
    features = torch.cat([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)

    perm = torch.randperm(features.size(0))
    return features[perm], targets[perm]


class GRUWithEncoders(nn.Module):

    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        self.price_indices = (
            list(range(0, 22))
            + list(range(44, 48))
            + list(range(52, 74))
            + list(range(96, 100))
        )
        self.vol_indices = (
            list(range(22, 44))
            + list(range(48, 52))
            + list(range(74, 96))
            + list(range(100, 104))
        )
        self.add_indices = list(range(104, 112))

        self.price_encoder = nn.Sequential(
            nn.Linear(len(self.price_indices), 64), nn.SiLU()
        )
        self.vol_encoder = nn.Sequential(
            nn.Linear(len(self.vol_indices), 64), nn.SiLU()
        )
        self.add_encoder = nn.Sequential(
            nn.Linear(len(self.add_indices), 32), nn.SiLU()
        )

        self.l1_encoder = nn.Sequential(nn.Linear(5, 16), nn.SiLU())

        self.gru = nn.GRU(
            input_size=176,  # 64 + 64 + 32 + 16
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def _extract_l1(self, p_b, v_b, p_a, v_a):
        best_bid_p, bid_idx = torch.max(p_b, dim=-1, keepdim=True)
        best_bid_v = torch.gather(v_b, dim=-1, index=bid_idx)

        best_ask_p, ask_idx = torch.min(p_a, dim=-1, keepdim=True)
        best_ask_v = torch.gather(v_a, dim=-1, index=ask_idx)

        spread = best_ask_p - best_bid_p
        mid = 0.5 * (best_bid_p + best_ask_p)

        vol_sum = best_bid_v + best_ask_v + 1e-6
        imbalance = (best_bid_v - best_ask_v) / vol_sum

        return spread, imbalance, mid

    def forward(self, x, h=None):
        p = self.price_encoder(x[:, :, self.price_indices])
        v = self.vol_encoder(x[:, :, self.vol_indices])
        a = self.add_encoder(x[:, :, self.add_indices])

        s0, imb0, mid0 = self._extract_l1(
            x[:, :, 0:11], x[:, :, 22:33], x[:, :, 11:22], x[:, :, 33:44]
        )
        s1, imb1, mid1 = self._extract_l1(
            x[:, :, 52:63], x[:, :, 74:85], x[:, :, 63:74], x[:, :, 85:96]
        )
        mid_diff = mid0 - mid1

        l1_raw = torch.cat([s0, imb0, s1, imb1, mid_diff], dim=-1)
        l1 = self.l1_encoder(l1_raw)

        combined = torch.cat([p, v, a, l1], dim=-1)

        out, h_next = self.gru(combined, h)

        pred = 2.0 * torch.tanh(self.head(out))

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUWithEncoders(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )


import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader(cfg, seed=None) -> DataLoader:
    ds = ParquetFastChunkDataset(cfg.train_path, chunk_size=cfg.chunk_size)
    
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=cfg.full_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=chunk_collate_fn,
    )