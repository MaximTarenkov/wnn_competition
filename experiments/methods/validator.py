import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

from utils import GlobalAccumulator, SEQUENCE_LENGTH, FEATURE_COLUMNS, TARGET_COLUMNS

VALID_PARQUET_PATH = "../datasets/valid.parquet"


class ParquetValDataset(Dataset):
    def __init__(self, parquet_path, sample_stride=1):
        self.parquet_path = parquet_path
        self.parquet = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.parquet.num_row_groups
        self.indices = list(range(0, self.num_row_groups, sample_stride))
        self.columns = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS) + ["need_prediction", "is_scored"]
        self.n_feat = len(FEATURE_COLUMNS)
        self.n_targ = len(TARGET_COLUMNS)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rg_idx = self.indices[idx]
        table = self.parquet.read_row_group(rg_idx, columns=self.columns, use_threads=False)

        feat = np.empty((SEQUENCE_LENGTH, self.n_feat), dtype=np.float32)
        for i, c in enumerate(FEATURE_COLUMNS):
            feat[:, i] = table[c].to_numpy(zero_copy_only=False)

        targ = np.empty((SEQUENCE_LENGTH, self.n_targ), dtype=np.float32)
        for i, c in enumerate(TARGET_COLUMNS):
            targ[:, i] = table[c].to_numpy(zero_copy_only=False)

        need = table["need_prediction"].to_numpy(zero_copy_only=False).astype(bool)
        scored = table["is_scored"].to_numpy(zero_copy_only=False).astype(bool)
        mask = need & scored

        return torch.from_numpy(feat), torch.from_numpy(targ), torch.from_numpy(mask)


@torch.inference_mode()
def evaluate(models, cfg, device, sample_stride=1, batch_size=4, num_workers=2):
    device = torch.device(device)
    use_cuda = device.type == "cuda"

    is_single = isinstance(models, torch.nn.Module)
    if is_single:
        models_dict = {"default": models}
    elif isinstance(models, dict):
        models_dict = models
    elif isinstance(models, (list, tuple)):
        models_dict = {f"model_{i}": m for i, m in enumerate(models)}
    else:
        raise ValueError("models must be nn.Module, dict or list of models")

    for m in models_dict.values():
        m.eval()

    valid_path = getattr(cfg, "valid_path", VALID_PARQUET_PATH)
    val_dataset = ParquetValDataset(valid_path, sample_stride=sample_stride)

    effective_workers = num_workers if len(val_dataset) > batch_size else 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=effective_workers,
        pin_memory=use_cuda,
        prefetch_factor=2 if effective_workers > 0 else None,
        persistent_workers=False,
    )

    accumulators = {name: GlobalAccumulator() for name in models_dict.keys()}

    for batch_features, batch_targets, batch_masks in val_loader:
        x = batch_features.to(device, non_blocking=use_cuda)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for name, model in models_dict.items():
                out = model(x)
                preds = out[0] if isinstance(out, tuple) else out
                preds_np = preds.float().cpu().numpy()

                targets_np = batch_targets.numpy()
                masks_np = batch_masks.numpy()

                for t, p, m in zip(targets_np, preds_np, masks_np):
                    accumulators[name].add(t, p, m)

    results = {name: acc.result() for name, acc in accumulators.items()}
    return results["default"] if is_single else results