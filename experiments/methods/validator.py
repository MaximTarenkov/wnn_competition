import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import GlobalAccumulator, SEQUENCE_LENGTH, N_FEATURES


class ValDataset(Dataset):
    def __init__(self, feat_path, targ_path, mask_path, sample_stride=1):
        self.feat_path = feat_path
        self.targ_path = targ_path
        self.mask_path = mask_path
        self.sample_stride = sample_stride
        
        bytes_per_seq = SEQUENCE_LENGTH * N_FEATURES * 4
        self.num_seq = os.path.getsize(feat_path) // bytes_per_seq
        self.indices = list(range(0, self.num_seq, sample_stride))

        self.features = None
        self.targets = None
        self.masks = None

    def _init_mmap(self):
        if self.features is None:
            self.features = np.memmap(
                self.feat_path, dtype=np.float32, mode="r",
                shape=(self.num_seq, SEQUENCE_LENGTH, N_FEATURES)
            )
            self.targets = np.memmap(
                self.targ_path, dtype=np.float32, mode="r",
                shape=(self.num_seq, SEQUENCE_LENGTH, 2)
            )
            self.masks = np.memmap(
                self.mask_path, dtype=bool, mode="r",
                shape=(self.num_seq, SEQUENCE_LENGTH)
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        self._init_mmap()
        seq_idx = self.indices[idx]

        return (
            torch.from_numpy(self.features[seq_idx].copy()),
            torch.from_numpy(self.targets[seq_idx].copy()),
            torch.from_numpy(self.masks[seq_idx].copy())
        )


@torch.inference_mode()
def evaluate(model, cfg, device, sample_stride=1, batch_size=8, num_workers=2):
    model.eval()
    device = torch.device(device)
    use_cuda = device.type == 'cuda'

    val_dataset = ValDataset(
        cfg.val_feat_mmap, 
        cfg.val_targ_mmap, 
        cfg.val_mask_mmap, 
        sample_stride=sample_stride
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=False
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
def evaluate_chunked(model, cfg, device, chunk_size=2000, sample_stride=1, batch_size=8, num_workers=2):
    model.eval()
    device = torch.device(device)
    use_cuda = device.type == 'cuda'

    val_dataset = ValDataset(cfg.val_feat_mmap, cfg.val_targ_mmap, cfg.val_mask_mmap, sample_stride=sample_stride)
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