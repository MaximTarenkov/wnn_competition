import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH


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
        features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)

        feat_tensor = torch.from_numpy(features).view(self.chunks_per_seq, self.chunk_size, -1)
        targ_tensor = torch.from_numpy(targets).view(self.chunks_per_seq, self.chunk_size, -1)

        return feat_tensor, targ_tensor


def chunk_collate_fn(batch):
    features = torch.cat([item[0] for item in batch], dim=0)
    targets = torch.cat([item[1] for item in batch], dim=0)

    perm = torch.randperm(features.size(0))
    return features[perm], targets[perm]


class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.sigmoid(self.fc2(x))
        gated = self.fc1(x) * gate
        return self.norm(x + self.dropout(gated))

class GRU_InputGated(nn.Module):

    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        self.input_gated_layers = nn.Sequential(
            GatedResidualBlock(input_dim), GatedResidualBlock(input_dim)
        )

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        x_gated = self.input_gated_layers(x)

        out, h_next = self.gru(x_gated, h)

        pred = 2.0 * torch.tanh(self.head(out))
        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRU_InputGated(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim
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