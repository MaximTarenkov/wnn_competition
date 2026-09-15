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

        self.price_encoder = nn.Sequential(nn.Linear(52, 64), nn.SiLU())
        self.vol_encoder = nn.Sequential(nn.Linear(52, 64), nn.SiLU())
        self.add_encoder = nn.Sequential(nn.Linear(8, 32), nn.SiLU())

        self.gru = nn.GRU(
            input_size=160,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def _process_book(self, p_b, v_b, p_a, v_a, dp, dv):
        p_b_safe = torch.where(
            v_b == 0.0, torch.full_like(p_b, -float("inf")), p_b
        )
        sorted_p_b, idx_b = torch.sort(p_b_safe, dim=-1, descending=True)
        sorted_v_b = torch.gather(v_b, dim=-1, index=idx_b)
        sorted_p_b = torch.where(
            torch.isinf(sorted_p_b), torch.zeros_like(sorted_p_b), sorted_p_b
        )

        p_a_safe = torch.where(
            v_a == 0.0, torch.full_like(p_a, float("inf")), p_a
        )
        sorted_p_a, idx_a = torch.sort(p_a_safe, dim=-1, descending=False)
        sorted_v_a = torch.gather(v_a, dim=-1, index=idx_a)
        sorted_p_a = torch.where(
            torch.isinf(sorted_p_a), torch.zeros_like(sorted_p_a), sorted_p_a
        )

        p_feats = torch.cat([sorted_p_b, sorted_p_a, dp], dim=-1)
        v_feats = torch.cat([sorted_v_b, sorted_v_a, dv], dim=-1)

        return p_feats, v_feats

    def forward(self, x, h=None):
        p0, v0 = self._process_book(
            x[:, :, 0:11],
            x[:, :, 22:33],
            x[:, :, 11:22],
            x[:, :, 33:44],
            x[:, :, 44:48],
            x[:, :, 48:52],
        )
        p1, v1 = self._process_book(
            x[:, :, 52:63],
            x[:, :, 74:85],
            x[:, :, 63:74],
            x[:, :, 85:96],
            x[:, :, 96:100],
            x[:, :, 100:104],
        )

        p_all = torch.cat([p0, p1], dim=-1)
        v_all = torch.cat([v0, v1], dim=-1)
        a_all = x[:, :, 104:112]

        p = self.price_encoder(p_all)
        v = self.vol_encoder(v_all)
        a = self.add_encoder(a_all)

        combined = torch.cat([p, v, a], dim=-1)

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