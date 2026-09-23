import torch
import torch.nn as nn


class GRUChampion(nn.Module):
    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        # 1. LOB-энкодеры стакана (22 -> 32)
        self.enc_p0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_p1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())

        # 2. Энкодеры трейдов: подаем вместе dp, dv и их произведение (4 + 4 + 4 = 12 -> 16)
        self.enc_trades0 = nn.Sequential(nn.Linear(12, 16), nn.SiLU())
        self.enc_trades1 = nn.Sequential(nn.Linear(12, 16), nn.SiLU())

        # 3. Aux-энкодер (8 -> 16)
        self.enc_aux = nn.Sequential(nn.Linear(8, 16), nn.SiLU())

        # 4. L1 + Spread Dynamics энкодер (7 фичей -> 24) с LayerNorm для стабильности
        # Фичи: [s0, imb0, s1, imb1, spread, spread_vel, basis_vol]
        self.enc_l1 = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, 24),
            nn.SiLU()
        )

        # Суммарная размерность входа в GRU:
        # 32*4 (128) + 16*2 (32) + 16 (aux) + 24 (l1_dyn) = 200
        self.gru = nn.GRU(
            input_size=200,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def _extract_l1_and_dynamics(self, p_b0, v_b0, p_a0, v_a0, p_b1, v_b1, p_a1, v_a1, B, T):
        # 1. Честные L1 по обоим инструментам через max / min
        best_bid_p0, b_idx0 = p_b0.max(dim=-1, keepdim=True)
        best_ask_p0, a_idx0 = p_a0.min(dim=-1, keepdim=True)
        best_bid_v0 = torch.gather(v_b0, dim=-1, index=b_idx0)
        best_ask_v0 = torch.gather(v_a0, dim=-1, index=a_idx0)

        best_bid_p1, b_idx1 = p_b1.max(dim=-1, keepdim=True)
        best_ask_p1, a_idx1 = p_a1.min(dim=-1, keepdim=True)
        best_bid_v1 = torch.gather(v_b1, dim=-1, index=b_idx1)
        best_ask_v1 = torch.gather(v_a1, dim=-1, index=a_idx1)

        # Метрики стакана
        s0 = best_ask_p0 - best_bid_p0
        mid0 = 0.5 * (best_bid_p0 + best_ask_p0)
        imb0 = best_bid_v0 - best_ask_v0

        s1 = best_ask_p1 - best_bid_p1
        mid1 = 0.5 * (best_bid_p1 + best_ask_p1)
        imb1 = best_bid_v1 - best_ask_v1

        # Главный межактивный базис
        spread = mid0 - mid1
        basis_vol = (best_bid_v0 + best_ask_v0) - (best_bid_v1 + best_ask_v1)

        # 2. Скорость спреда во времени: d(Spread)/dt
        spread_seq = spread.view(B, T, 1)
        # Каузальная разность во времени (первый шаг заполняем нулем)
        spread_vel = torch.cat([
            torch.zeros_like(spread_seq[:, :1]),
            spread_seq[:, 1:] - spread_seq[:, :-1]
        ], dim=1).view(B * T, 1)

        l1_features = torch.cat([s0, imb0, s1, imb1, spread, spread_vel, basis_vol], dim=-1)
        return l1_features

    def forward(self, x, h=None):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # Стакан i0 и i1
        p_b0, p_a0 = x_flat[:, 0:11], x_flat[:, 11:22]
        v_b0, v_a0 = x_flat[:, 22:33], x_flat[:, 33:44]
        p_b1, p_a1 = x_flat[:, 52:63], x_flat[:, 63:74]
        v_b1, v_a1 = x_flat[:, 74:85], x_flat[:, 85:96]

        # Трейды i0 и i1
        dp0, dv0 = x_flat[:, 44:48], x_flat[:, 48:52]
        dp1, dv1 = x_flat[:, 96:100], x_flat[:, 100:104]

        # 1. Энкодинг стакана
        e_p0 = self.enc_p0_lob(x_flat[:, 0:22])
        e_v0 = self.enc_v0_lob(x_flat[:, 22:44])
        e_p1 = self.enc_p1_lob(x_flat[:, 52:74])
        e_v1 = self.enc_v1_lob(x_flat[:, 74:96])

        # 2. Энкодинг трейдов с учетом денежного потока (dp * dv)
        tr0_combined = torch.cat([dp0, dv0, dp0 * dv0], dim=-1)
        tr1_combined = torch.cat([dp1, dv1, dp1 * dv1], dim=-1)
        e_tr0 = self.enc_trades0(tr0_combined)
        e_tr1 = self.enc_trades1(tr1_combined)

        # 3. Aux фичи
        e_aux = self.enc_aux(x_flat[:, 104:112])

        # 4. L1 + динамика спреда
        l1_raw = self._extract_l1_and_dynamics(p_b0, v_b0, p_a0, v_a0, p_b1, v_b1, p_a1, v_a1, B, T)
        e_l1 = self.enc_l1(l1_raw)

        # 5. Сборка (размерность 200)
        combined = torch.cat([
            e_p0, e_v0, e_p1, e_v1,
            e_tr0, e_tr1,
            e_aux, e_l1
        ], dim=-1).view(B, T, 200).to(x.dtype)

        out, h_next = self.gru(combined, h)
        pred = 2.0 * torch.tanh(self.head(out))

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUChampion(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )