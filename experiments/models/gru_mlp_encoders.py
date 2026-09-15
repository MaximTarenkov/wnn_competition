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

        self.gru = nn.GRU(
            input_size=160,  # 64 + 64 + 32
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        p = self.price_encoder(x[:, :, self.price_indices])
        v = self.vol_encoder(x[:, :, self.vol_indices])
        a = self.add_encoder(x[:, :, self.add_indices])

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


def get_dataloader(cfg) -> DataLoader:
    ds = ParquetFastChunkDataset(cfg.train_path, chunk_size=cfg.chunk_size)
    return DataLoader(
        ds,
        batch_size=cfg.full_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=chunk_collate_fn,
    )