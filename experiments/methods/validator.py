import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from utils import BlockAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS


class ParquetValDataset(Dataset):
    def __init__(self, parquet_path, sample_stride=1):
        self.parquet_path = parquet_path
        parquet_file = pq.ParquetFile(parquet_path)
        self.indices = list(range(0, parquet_file.num_row_groups, sample_stride))
        self.parquet = None
        self.load_cols = FEATURE_COLUMNS + list(TARGET_COLUMNS) + ['need_prediction', 'is_scored']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.parquet is None:
            self.parquet = pq.ParquetFile(self.parquet_path)

        table = self.parquet.read_row_group(self.indices[idx], columns=self.load_cols)

        features = table.select(FEATURE_COLUMNS).to_pandas().to_numpy(dtype=np.float32).copy()
        targets = table.select(TARGET_COLUMNS).to_pandas().to_numpy(dtype=np.float32).copy()

        is_scored = table['is_scored'].to_numpy().astype(bool)
        need_pred = table['need_prediction'].to_numpy().astype(bool)
        mask = is_scored & need_pred

        return (
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.from_numpy(mask)
        )


@torch.inference_mode()
def evaluate(model, parquet_path, device, sample_stride=1, batch_size=8, num_workers=2):
    model.eval()
    device = torch.device(device)
    use_cuda = device.type == 'cuda'

    val_dataset = ParquetValDataset(parquet_path, sample_stride=sample_stride)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )

    accumulator = BlockAccumulator()

    for batch_features, batch_targets, batch_masks in val_loader:
        x = batch_features.to(device, non_blocking=True)

        out = model(x)
        preds = out[0] if isinstance(out, tuple) else out

        preds_cpu = preds.detach().cpu().numpy()
        targets_cpu = batch_targets.numpy()
        masks_cpu = batch_masks.numpy()

        for t, p, m in zip(targets_cpu, preds_cpu, masks_cpu):
            accumulator.add(t, p, m)

    return accumulator.result()