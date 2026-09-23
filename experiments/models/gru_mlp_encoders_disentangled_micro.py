import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUWithDisentangledMicro(nn.Module):
    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        # 4 x LOB-энкодеры (22 -> 32)
        self.enc_p0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_p1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())

        # 4 x Extra-энкодеры (4 -> 8)
        self.enc_p0_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_v0_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_p1_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_v1_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())

        # 1 x Aux-энкодер (8 -> 16)
        self.enc_aux = nn.Sequential(nn.Linear(8, 16), nn.SiLU())

        # 1 x Базовый L1-энкодер (5 -> 16)
        self.enc_l1 = nn.Sequential(nn.Linear(5, 16), nn.SiLU())

        # 1 x Микроструктурный энкодер с LayerNorm на входе (16 -> 32)
        self.enc_micro = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, 32),
            nn.SiLU()
        )

        # 32*4 (LOB) + 8*4 (Extra) + 16 (Aux) + 16 (L1) + 32 (Micro) = 224
        self.gru = nn.GRU(
            input_size=224,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def _extract_l1_2d(self, p_b, v_b, p_a, v_a):
        # Линейные операции инвариантны к сдвигу и масштабу нормализации
        best_bid_p = p_b[:, 0:1]
        best_ask_p = p_a[:, 0:1]
        best_bid_v = v_b[:, 0:1]
        best_ask_v = v_a[:, 0:1]

        spread = best_ask_p - best_bid_p
        mid = 0.5 * (best_bid_p + best_ask_p)
        vol_imbalance = best_bid_v - best_ask_v

        return spread, vol_imbalance, mid

    def _compute_micro_features(self, p_b0, v_b0, p_a0, v_a0, p_b1, v_b1, p_a1, v_a1, eps=1e-6):
        def _asset_stats(p_b, v_b, p_a, v_a):
            # Переводим нормализованные объемы в строго положительные веса
            v_b_pos = F.softplus(v_b)
            v_a_pos = F.softplus(v_a)

            # Взвешенные произведения (Micro-price cross weighting)
            cross_pv = p_b * v_a_pos + p_a * v_b_pos
            sum_v = v_b_pos + v_a_pos

            wap1 = cross_pv[:, :1].sum(dim=-1, keepdim=True) / (sum_v[:, :1].sum(dim=-1, keepdim=True) + eps)
            wap2 = cross_pv[:, :2].sum(dim=-1, keepdim=True) / (sum_v[:, :2].sum(dim=-1, keepdim=True) + eps)
            wap3 = cross_pv[:, :3].sum(dim=-1, keepdim=True) / (sum_v[:, :3].sum(dim=-1, keepdim=True) + eps)
            wap5 = cross_pv[:, :5].sum(dim=-1, keepdim=True) / (sum_v[:, :5].sum(dim=-1, keepdim=True) + eps)
            wap_diff = wap5 - wap1

            # Кумулятивный дисбаланс объемов на положительных весах (диапазон 0..1)
            imb5 = v_b_pos[:, :5].sum(dim=-1, keepdim=True) / (sum_v[:, :5].sum(dim=-1, keepdim=True) + eps)
            imball = v_b_pos.sum(dim=-1, keepdim=True) / (sum_v.sum(dim=-1, keepdim=True) + eps)

            return wap1, wap2, wap3, wap5, wap_diff, imb5, imball

        w1_0, w2_0, w3_0, w5_0, wdiff_0, imb5_0, imball_0 = _asset_stats(p_b0, v_b0, p_a0, v_a0)
        w1_1, w2_1, w3_1, w5_1, wdiff_1, imb5_1, imball_1 = _asset_stats(p_b1, v_b1, p_a1, v_a1)

        # Межактивный базис WAP
        basis_1 = w1_0 - w1_1
        basis_5 = w5_0 - w5_1

        return torch.cat([
            w1_0, w2_0, w3_0, w5_0, wdiff_0, imb5_0, imball_0,
            w1_1, w2_1, w3_1, w5_1, wdiff_1, imb5_1, imball_1,
            basis_1, basis_5
        ], dim=-1)

    def forward(self, x, h=None):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        p_b0, p_a0 = x_flat[:, 0:11], x_flat[:, 11:22]
        v_b0, v_a0 = x_flat[:, 22:33], x_flat[:, 33:44]

        p_b1, p_a1 = x_flat[:, 52:63], x_flat[:, 63:74]
        v_b1, v_a1 = x_flat[:, 74:85], x_flat[:, 85:96]

        # 1. Прогон через изолированные энкодеры
        e_p0_lob = self.enc_p0_lob(x_flat[:, 0:22])
        e_v0_lob = self.enc_v0_lob(x_flat[:, 22:44])
        e_p0_ext = self.enc_p0_ext(x_flat[:, 44:48])
        e_v0_ext = self.enc_v0_ext(x_flat[:, 48:52])

        e_p1_lob = self.enc_p1_lob(x_flat[:, 52:74])
        e_v1_lob = self.enc_v1_lob(x_flat[:, 74:96])
        e_p1_ext = self.enc_p1_ext(x_flat[:, 96:100])
        e_v1_ext = self.enc_v1_ext(x_flat[:, 100:104])

        e_aux = self.enc_aux(x_flat[:, 104:112])

        # 2. Базовый L1 срез
        s0, imb0, mid0 = self._extract_l1_2d(p_b0, v_b0, p_a0, v_a0)
        s1, imb1, mid1 = self._extract_l1_2d(p_b1, v_b1, p_a1, v_a1)
        mid_diff = mid0 - mid1
        e_l1 = self.enc_l1(torch.cat([s0, imb0, s1, imb1, mid_diff], dim=-1))

        # 3. Адаптированный микроструктурный срез (16 фичей)
        micro_raw = self._compute_micro_features(p_b0, v_b0, p_a0, v_a0, p_b1, v_b1, p_a1, v_a1)
        e_micro = self.enc_micro(micro_raw)

        # 4. Конкатенация (dim=224)
        combined = torch.cat([
            e_p0_lob, e_v0_lob, e_p0_ext, e_v0_ext,
            e_p1_lob, e_v1_lob, e_p1_ext, e_v1_ext,
            e_aux, e_l1, e_micro
        ], dim=-1).view(B, T, 224).to(x.dtype)

        out, h_next = self.gru(combined, h)
        pred = 2.0 * torch.tanh(self.head(out))

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUWithDisentangledMicro(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )