import numpy as np
import pyarrow.parquet as pq
import torch
from utils import BlockAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS


def evaluate(model, parquet_path, device, sample_stride=1):
    model.eval()
    val_parquet = pq.ParquetFile(parquet_path)
    accumulator = BlockAccumulator()
    num_sequences = val_parquet.num_row_groups
    sampled_indices = list(range(0, num_sequences, sample_stride))

    load_cols = FEATURE_COLUMNS + list(TARGET_COLUMNS) + ['need_prediction', 'is_scored']

    with torch.no_grad():
        for i in sampled_indices:
            table = val_parquet.read_row_group(i, columns=load_cols)

            features = table.select(FEATURE_COLUMNS).to_pandas().to_numpy(dtype=np.float32)
            targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
            mask = table['is_scored'].to_numpy().astype(bool) & table['need_prediction'].to_numpy().astype(bool)

            x = torch.from_numpy(features).unsqueeze(0).to(device)
            out = model(x)
            preds = out[0] if isinstance(out, tuple) else out

            accumulator.add(targets, preds.squeeze(0).cpu().numpy(), mask)

    return accumulator.result()
