import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUTwoStreamLateFusion(nn.Module):
    """
    Two-Stream Архитектура (Modular Disentanglement):
      - Stream 1 (State GRU): изолированно учит геометрию стакана (192 -> 128)
      - Stream 2 (Delta GRU): изолированно учит поток импульсов (32 -> 64)
      - Late Fusion: конкатенация скрытых состояний [h_state, h_delta] перед Head
    """
    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        self.state_hidden_dim = hidden_dim       # 128
        self.delta_hidden_dim = hidden_dim // 2  # 64

        # =====================================================================
        # STREAM 1: Энкодеры стакана + State GRU
        # =====================================================================
        self.enc_p0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_p1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())

        self.enc_p0_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_v0_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_p1_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_v1_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())

        self.enc_aux = nn.Sequential(nn.Linear(8, 16), nn.SiLU())
        self.enc_l1 = nn.Sequential(nn.Linear(5, 16), nn.SiLU())

        self.state_gru = nn.GRU(
            input_size=192,
            hidden_size=self.state_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # =====================================================================
        # STREAM 2: Проектор дельт + Delta GRU
        # =====================================================================
        self.delta_proj = nn.Linear(input_dim, 32, bias=False)

        self.delta_gru = nn.GRU(
            input_size=32,
            hidden_size=self.delta_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # =====================================================================
        # LATE FUSION HEAD: Принимает решения на основе обоих миров
        # =====================================================================
        # 128 (State memory) + 64 (Delta memory) = 192
        fused_dim = self.state_hidden_dim + self.delta_hidden_dim
        
        # 2-слойный MLP-арбитр для нелинейного взвешивания мнений сетей
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

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

        # Распаковка h, если передано кортежем или срезом
        if h is not None:
            h_state = h[:, :, :self.state_hidden_dim].contiguous()
            h_delta = h[:, :, self.state_hidden_dim:].contiguous()
        else:
            h_state, h_delta = None, None

        # 1. ПРОГОН СТРИМА СТАКАНА (State Stream)
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
            x_flat[:, 0:11], x_flat[:, 22:33],
            x_flat[:, 11:22], x_flat[:, 33:44]
        )
        s1, imb1, mid1 = self._extract_l1_2d(
            x_flat[:, 52:63], x_flat[:, 74:85],
            x_flat[:, 63:74], x_flat[:, 85:96]
        )
        mid_diff = mid0 - mid1

        l1_raw = torch.cat([s0, imb0, s1, imb1, mid_diff], dim=-1)
        e_l1 = self.enc_l1(l1_raw)

        combined_state = torch.cat([
            e_p0_lob, e_v0_lob, e_p0_ext, e_v0_ext,
            e_p1_lob, e_v1_lob, e_p1_ext, e_v1_ext,
            e_aux, e_l1
        ], dim=-1).view(B, T, 192).to(x.dtype)

        out_state, h_state_next = self.state_gru(combined_state, h_state)

        # 2. ПРОГОН СТРИМА ИМПУЛЬСОВ (Delta Stream)
        d_proj = self.delta_proj(x_flat).view(B, T, 32)
        d_diff = torch.cat([
            torch.zeros_like(d_proj[:, :1, :]),
            d_proj[:, 1:, :] - d_proj[:, :-1, :]
        ], dim=1)
        e_delta = F.silu(d_diff).to(x.dtype)

        out_delta, h_delta_next = self.delta_gru(e_delta, h_delta)

        # 3. LATE FUSION (Конкатенация представлений)
        fused = torch.cat([out_state, out_delta], dim=-1) # [B, T, 192]
        pred = 2.0 * torch.tanh(self.head(fused))

        # Упаковываем скрытое состояние для совместимости
        h_next = torch.cat([h_state_next, h_delta_next], dim=-1)

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUTwoStreamLateFusion(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )