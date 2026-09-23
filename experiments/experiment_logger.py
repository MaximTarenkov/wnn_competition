import csv
from datetime import datetime
import logging
import os
from pathlib import Path


class ExperimentLogger:
    def __init__(self, base_dir, exp_name, seed):
        seed_dir = Path(base_dir) / exp_name / f"seed_{seed}"
        if seed_dir.exists():
            counter = 1
            while (Path(base_dir) / exp_name / f"seed_{seed}_{counter}").exists():
                counter += 1
            seed_dir = Path(base_dir) / exp_name / f"seed_{seed}_{counter}"

        seed_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = str(seed_dir)

        self.logger = logging.getLogger(self.run_dir)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(os.path.join(self.run_dir, "run.log"), encoding="utf-8")
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.csv_path = os.path.join(self.run_dir, "history.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "epoch", "step", "event_type", "lr",
                "train_loss", "train_wp", "val_wp", "t0", "t1", "patience"
            ])

    def info(self, msg):
        self.logger.info(msg)

    def log_train_step(self, epoch, step, lr, train_loss, train_wp):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now_str, epoch, step, "train_step", f"{lr:.6e}",
                f"{train_loss:.6f}", f"{train_wp:.6f}", "", "", "", ""
            ])

    def log_val_event(self, epoch, step, event_type, lr, res, patience):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                now_str, epoch, step, event_type, f"{lr:.6e}",
                "", "", f"{res['weighted_pearson']:.6f}",
                f"{res['t0']:.6f}", f"{res['t1']:.6f}", patience
            ])