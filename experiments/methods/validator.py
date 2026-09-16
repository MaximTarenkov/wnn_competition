import numpy as np
import pyarrow.feather as feather
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from utils import GlobalAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS


class ValDataset(Dataset):
    def __init__(self, feather_path, parquet_meta_path, sample_stride=1):
        self.table = feather.read_table(feather_path, memory_map=True)

        parquet_file = pq.ParquetFile(parquet_meta_path)
        num_rg = parquet_file.num_row_groups
        
        row_counts = [parquet_file.metadata.row_group(i).num_rows for i in range(num_rg)]
        
        self.offsets = np.cumsum([0] + row_counts)
        self.indices = list(range(0, num_rg, sample_stride))

        self.feature_cols = FEATURE_COLUMNS
        self.target_cols = list(TARGET_COLUMNS)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rg_idx = self.indices[idx]
        start_row = self.offsets[rg_idx]
        length = self.offsets[rg_idx + 1] - start_row

        sub_table = self.table.slice(start_row, length)

        features = sub_table.select(self.feature_cols).to_pandas().to_numpy(dtype=np.float32)
        targets = sub_table.select(self.target_cols).to_pandas().to_numpy(dtype=np.float32)

        is_scored = sub_table['is_scored'].to_numpy().astype(bool)
        need_pred = sub_table['need_prediction'].to_numpy().astype(bool)
        mask = is_scored & need_pred

        return (
            torch.from_numpy(features),
            torch.from_numpy(targets),
            torch.from_numpy(mask)
        )


@torch.inference_mode()
def evaluate(model, feather_path, parquet_meta_path, device, sample_stride=1, batch_size=8, num_workers=2):
    model.eval()
    device = torch.device(device)
    use_cuda = device.type == 'cuda'

    val_dataset = ValDataset(feather_path, parquet_meta_path, sample_stride=sample_stride)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0)
    )

    accumulator = GlobalAccumulator()

    for batch_features, batch_targets, batch_masks in val_loader:
        x = batch_features.to(device, non_blocking=use_cuda)

        out = model(x)
        preds = out[0] if isinstance(out, tuple) else out

        preds_cpu = preds.detach().cpu().numpy()
        targets_cpu = batch_targets.numpy()
        masks_cpu = batch_masks.numpy()

        for t, p, m in zip(targets_cpu, preds_cpu, masks_cpu):
            accumulator.add(t, p, m)

    return accumulator.result()


@torch.inference_mode()
def evaluate_chunked(model, feather_path, parquet_meta_path, device, chunk_size=2000, sample_stride=1, batch_size=8, num_workers=2):
    model.eval()
    device = torch.device(device)
    use_cuda = device.type == 'cuda'

    val_dataset = ValDataset(feather_path, parquet_meta_path, sample_stride=sample_stride)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0)
    )

    accumulator = GlobalAccumulator()

    for batch_features, batch_targets, batch_masks in val_loader:
        B, T, D = batch_features.shape
        h = None
        all_preds = []

        for start_idx in range(0, T, chunk_size):
            end_idx = min(start_idx + chunk_size, T)
            chunk_x = batch_features[:, start_idx:end_idx, :].to(device, non_blocking=use_cuda)

            out = model(chunk_x, h)
            if isinstance(out, tuple):
                preds_chunk, h = out[0], out[1]
            else:
                preds_chunk, h = out, None

            all_preds.append(preds_chunk.detach().cpu())

        preds_cpu = torch.cat(all_preds, dim=1).numpy()
        targets_cpu = batch_targets.numpy()
        masks_cpu = batch_masks.numpy()

        for t, p, m in zip(targets_cpu, preds_cpu, masks_cpu):
            accumulator.add(t, p, m)

    return accumulator.result()