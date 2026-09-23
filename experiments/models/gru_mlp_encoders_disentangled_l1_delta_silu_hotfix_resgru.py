import torch
import torch.nn as nn


class GRUDualStreamL1FixResGRU(nn.Module):
    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        # =====================================================================
        # 1. 100% ОРИГИНАЛЬНЫЕ ЭНКОДЕРЫ ИЗ hotfix (БЕЗ ИЗМЕНЕНИЙ)
        # =====================================================================
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

        # 1 x Синтетический L1-энкодер (5 -> 16)
        self.enc_l1 = nn.Sequential(nn.Linear(5, 16), nn.SiLU())

        # 1 x Delta Stream (112 -> 32) с tanh (как в hotfix)
        self.delta_proj = nn.Linear(input_dim, 32, bias=False)
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=0.01)

        # =====================================================================
        # 2. ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ: RESIDUAL GRU (СЛОЙ 1 + СЛОЙ 2)
        # =====================================================================
        # Слой 1: принимает эмбеддинг стакана (224) -> выдает hidden_dim (128)
        self.gru1 = nn.GRU(
            input_size=224,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Слой 2: принимает выход первого слоя (128) -> выдает hidden_dim (128)
        self.gru2 = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def _extract_l1_2d(self, p_b, v_b, p_a, v_a):
        best_bid_p, bid_idx = p_b.max(dim=-1, keepdim=True)
        best_ask_p, ask_idx = p_a.min(dim=-1, keepdim=True)

        best_bid_v = torch.gather(v_b, dim=-1, index=bid_idx)
        best_ask_v = torch.gather(v_a, dim=-1, index=ask_idx)

        spread = best_ask_p - best_bid_p
        mid = 0.5 * (best_bid_p + best_ask_p)
        vol_imbalance = best_bid_v - best_ask_v

        return spread, vol_imbalance, mid

    def forward(self, x, h=None):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # 1. Прогон стакана (как в hotfix)
        e_p0_lob = self.enc_p0_lob(x_flat[:, 0:22])
        e_v0_lob = self.enc_v0_lob(x_flat[:, 22:44])
        e_p0_ext = self.enc_p0_ext(x_flat[:, 44:48])
        e_v0_ext = self.enc_v0_ext(x_flat[:, 48:52])

        e_p1_lob = self.enc_p1_lob(x_flat[:, 52:74])
        e_v1_lob = self.enc_v1_lob(x_flat[:, 74:96])
        e_p1_ext = self.enc_p1_ext(x_flat[:, 96:100])
        e_v1_ext = self.enc_v1_ext(x_flat[:, 100:104])

        e_aux = self.enc_aux(x_flat[:, 104:112])

        s0, imb0, mid0 = self._extract_l1_2d(
            x_flat[:, 0:11],
            x_flat[:, 22:33],
            x_flat[:, 11:22],
            x_flat[:, 33:44]
        )
        s1, imb1, mid1 = self._extract_l1_2d(
            x_flat[:, 52:63],
            x_flat[:, 74:85],
            x_flat[:, 63:74],
            x_flat[:, 85:96]
        )
        mid_diff = mid0 - mid1

        l1_raw = torch.cat([s0, imb0, s1, imb1, mid_diff], dim=-1)
        e_l1 = self.enc_l1(l1_raw)

        # 2. Delta Stream (все 112 признаков с tanh)
        d_proj = self.delta_proj(x_flat).view(B, T, 32)
        d_diff = torch.cat([
            torch.zeros_like(d_proj[:, :1, :]),
            d_proj[:, 1:, :] - d_proj[:, :-1, :]
        ], dim=1)
        e_delta = torch.tanh(d_diff).reshape(B * T, 32)

        # 3. Сборка вектора (224)
        combined = torch.cat([
            e_p0_lob, e_v0_lob, e_p0_ext, e_v0_ext,
            e_p1_lob, e_v1_lob, e_p1_ext, e_v1_ext,
            e_aux, e_l1, e_delta
        ], dim=-1).view(B, T, 224).to(x.dtype)

        # 4. RESIDUAL ПРОГОН СКВОЗЬ РЕКУРРЕНТНЫЕ СЛОИ
        if h is not None:
            h1 = h[0:1].contiguous()
            h2 = h[1:2].contiguous()
        else:
            h1, h2 = None, None

        out1, h1_next = self.gru1(combined, h1)
        out2, h2_next = self.gru2(out1, h2)

        # Skip-connection: выход слоя 1 напрямую суммируется с выходом слоя 2
        out = out1 + out2

        # Собираем скрытые состояния обратно в (2, B, hidden_dim) для полной совместимости с валидатором
        h_next = torch.cat([h1_next, h2_next], dim=0)

        pred = 2.0 * torch.tanh(self.head(out))

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUDualStreamL1FixResGRU(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )