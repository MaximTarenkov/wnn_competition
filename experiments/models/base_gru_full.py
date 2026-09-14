import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import FEATURE_COLUMNS, TARGET_COLUMNS


class FullParquetDataset(Dataset):
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        parquet_file = pq.ParquetFile(parquet_path)
        self.num_sequences = parquet_file.num_row_groups
        self.parquet = None
        self.load_columns = FEATURE_COLUMNS + list(TARGET_COLUMNS)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.parquet is None:
            self.parquet = pq.ParquetFile(self.parquet_path)

        table = self.parquet.read_row_group(idx, columns=self.load_columns)
        features = table.select(FEATURE_COLUMNS).to_pandas().to_numpy(dtype=np.float32)
        targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)

        return torch.from_numpy(features), torch.from_numpy(targets)


class BaselineGRU(nn.Module):
    def __init__(self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        out, h_next = self.gru(x, h)
        pred = 2.0 * torch.tanh(self.head(out))
        return pred, h_next


def create_model(cfg) -> nn.Module:
    return BaselineGRU(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim
    )


def get_dataloader(cfg) -> DataLoader:
    ds = FullParquetDataset(cfg.train_path)
    return DataLoader(ds, batch_size=cfg.full_batch_size, shuffle=True, num_workers=2, pin_memory=True)
