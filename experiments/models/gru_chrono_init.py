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


class BaselineGRU(nn.Module):

    def __init__(
        self,
        input_dim=112,
        hidden_dim=128,
        num_layers=2,
        output_dim=2,
        t_max=150.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        with torch.no_grad():
            for name, param in self.gru.named_parameters(): # По канонам Chrono init инициализируем только первый слой. Другие слои инициализировать нет смысла, модель хуже показывает себя. В общем, по метрикам даже так инициализация не лучше байзлайнового.
                if "bias_ih_l0" in name:
                    h_dim = param.size(0) // 3
                    param[0:h_dim].fill_(2.5)

                    log_t = torch.empty(h_dim).uniform_(0.0, np.log(t_max))
                    t = torch.exp(log_t)
                    param[h_dim : 2 * h_dim].copy_(
                        torch.log(torch.clamp(t - 1.0, min=1e-4))
                    )

                    param[2 * h_dim :].fill_(0.0)
                elif "bias_hh_l0" in name:
                    param.fill_(0.0)

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