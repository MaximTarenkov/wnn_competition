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


class GatedLinearUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2)

    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, group_dims, emb_dim=32):
        super().__init__()
        self.num_groups = len(group_dims)
        self.emb_dim = emb_dim

        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.SiLU()
            ) for dim in group_dims
        ])

        self.flattened_dim = self.num_groups * emb_dim
        self.gating = nn.Sequential(
            GatedLinearUnit(self.flattened_dim, emb_dim),
            nn.Linear(emb_dim, self.num_groups),
            nn.Softmax(dim=-1)
        )

    def forward(self, group_inputs):
        embedded = [emb(x) for emb, x in zip(self.embeddings, group_inputs)]
        stacked = torch.stack(embedded, dim=2)
        flat_context = stacked.flatten(start_dim=2)
        weights = self.gating(flat_context).unsqueeze(-1)
        weighted = (stacked * weights).flatten(start_dim=2)
        return weighted


class VGRU(nn.Module):
    def __init__(self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()

        self.i0_p_idx = list(range(0, 22)) + list(range(44, 48))
        self.i0_v_idx = list(range(22, 44)) + list(range(48, 52))
        self.i1_p_idx = list(range(52, 74)) + list(range(96, 100))
        self.i1_v_idx = list(range(74, 96)) + list(range(100, 104))
        self.add_idx = list(range(104, 112))

        group_dims = [26, 26, 26, 26, 8]
        emb_dim = 32

        self.vsn = VariableSelectionNetwork(group_dims, emb_dim=emb_dim)
        vsn_output_dim = len(group_dims) * emb_dim

        self.gru = nn.GRU(
            input_size=vsn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        groups = [
            x[:, :, self.i0_p_idx],
            x[:, :, self.i0_v_idx],
            x[:, :, self.i1_p_idx],
            x[:, :, self.i1_v_idx],
            x[:, :, self.add_idx]
        ]

        vsn_features = self.vsn(groups)
        out, h_next = self.gru(vsn_features, h)
        pred = 2.0 * torch.tanh(self.head(out))
        return pred, h_next


def create_model(cfg) -> nn.Module:
    return VGRU(
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