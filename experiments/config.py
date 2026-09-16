from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    train_path: str = "../datasets/train.parquet"
    valid_path: str = "../datasets/valid.parquet"

    train_feat_mmap: str = "../datasets/train_features.mmap"
    train_targ_mmap: str = "../datasets/train_targets.mmap"

    runs_dir: str = "runs"

    seeds: List[int] = field(default_factory=lambda: [22252, 3327]) # [1, 2, 3, 4, 5, 6, 67, 7, 8, 9] # , 15762, 5575

    max_epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    log_interval: int = 25
    device: str = "cuda"

    input_dim: int = 112
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 2

    chunk_size: int = 2000
    full_batch_size: int = 3