import os
import numpy as np
import torch
import xgboost as xgb


class XGBDualModel:
    def __init__(
        self,
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "hist",
        device: str = "cuda",
        early_stopping_rounds: int = 50,
        add_deltas: bool = True,
    ):
        self.add_deltas = add_deltas
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": tree_method,
            "device": device,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_metric": "rmse",
            "random_state": 42,
        }
        self.model_0 = xgb.XGBRegressor(**params)
        self.model_1 = xgb.XGBRegressor(**params)

    # --- Заглушки для совместимости с PyTorch-пайплайном ---
    def eval(self):
        pass

    def train(self, mode=True):
        pass

    def to(self, device):
        return self
    # --------------------------------------------------------

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        if not self.add_deltas:
            return x
        delta = np.zeros_like(x)
        delta[1:] = x[1:] - x[:-1]
        return np.concatenate([x, delta], axis=-1)

    def fit(self, x_train, y_train, w_train, x_val, y_val):
        self.model_0.fit(
            x_train,
            y_train[:, 0],
            sample_weight=w_train[:, 0],
            eval_set=[(x_val, y_val[:, 0])],
            verbose=100,
        )
        self.model_1.fit(
            x_train,
            y_train[:, 1],
            sample_weight=w_train[:, 1],
            eval_set=[(x_val, y_val[:, 1])],
            verbose=100,
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        p0 = self.model_0.predict(x)
        p1 = self.model_1.predict(x)
        return np.clip(np.column_stack([p0, p1]), -2.0, 2.0)

    def __call__(self, x, h=None):
        is_torch = isinstance(x, torch.Tensor)
        dev = x.device if is_torch else None
        dtype = x.dtype if is_torch else None
        x_np = x.detach().cpu().numpy() if is_torch else x

        b_dim, t_dim, _ = x_np.shape
        preds = np.empty((b_dim, t_dim, 2), dtype=np.float32)

        for b in range(b_dim):
            feat = self.transform_features(x_np[b])
            preds[b] = self.predict(feat)

        if is_torch:
            return torch.from_numpy(preds).to(device=dev, dtype=dtype), None
        return preds, None

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        self.model_0.get_booster().save_model(os.path.join(dir_path, "xgb_t0.json"))
        self.model_1.get_booster().save_model(os.path.join(dir_path, "xgb_t1.json"))

    def load(self, dir_path: str):
        self.model_0.load_model(os.path.join(dir_path, "xgb_t0.json"))
        self.model_1.load_model(os.path.join(dir_path, "xgb_t1.json"))


def create_model(cfg) -> XGBDualModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return XGBDualModel(
        n_estimators=getattr(cfg, "xgb_n_estimators", 600),
        learning_rate=getattr(cfg, "xgb_lr", 0.05),
        max_depth=getattr(cfg, "xgb_max_depth", 6),
        subsample=getattr(cfg, "xgb_subsample", 0.8),
        colsample_bytree=getattr(cfg, "xgb_colsample_bytree", 0.8),
        device=device,
        add_deltas=getattr(cfg, "xgb_add_deltas", True),
    )