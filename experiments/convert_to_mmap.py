import os
import numpy as np
import pyarrow.parquet as pq
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH


def convert(parquet_path: str, feat_mmap_path: str, targ_mmap_path: str):
    pf = pq.ParquetFile(parquet_path)
    num_seq = pf.num_row_groups
    n_feat = len(FEATURE_COLUMNS)
    n_targ = len(TARGET_COLUMNS)

    os.makedirs(os.path.dirname(feat_mmap_path), exist_ok=True)
    os.makedirs(os.path.dirname(targ_mmap_path), exist_ok=True)

    feat_mmap = np.memmap(
        feat_mmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_seq, SEQUENCE_LENGTH, n_feat)
    )
    targ_mmap = np.memmap(
        targ_mmap_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_seq, SEQUENCE_LENGTH, n_targ)
    )

    columns = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)

    for i in range(num_seq):
        table = pf.read_row_group(i, columns=columns)
        f_arr = np.empty((SEQUENCE_LENGTH, n_feat), dtype=np.float32)
        for c_idx, c in enumerate(FEATURE_COLUMNS):
            f_arr[:, c_idx] = table[c].to_numpy(zero_copy_only=False)

        t_arr = np.empty((SEQUENCE_LENGTH, n_targ), dtype=np.float32)
        for c_idx, c in enumerate(TARGET_COLUMNS):
            t_arr[:, c_idx] = table[c].to_numpy(zero_copy_only=False)

        feat_mmap[i] = f_arr
        targ_mmap[i] = t_arr

    feat_mmap.flush()
    targ_mmap.flush()


if __name__ == "__main__":
    convert(
        "../datasets/train.parquet",
        "../datasets/train_features.mmap",
        "../datasets/train_targets.mmap"
    )