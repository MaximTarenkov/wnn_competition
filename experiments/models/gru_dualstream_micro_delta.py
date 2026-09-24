import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDualStreamMicroDelta(nn.Module):
    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        self.state_hidden_dim = hidden_dim
        self.delta_hidden_dim = hidden_dim // 2

        self.enc_p0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v0_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_p1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())
        self.enc_v1_lob = nn.Sequential(nn.Linear(22, 32), nn.SiLU())

        self.enc_p0_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_v0_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_p1_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.enc_v1_ext = nn.Sequential(nn.Linear(4, 8), nn.SiLU())

        self.enc_aux = nn.Sequential(nn.Linear(8, 16), nn.SiLU())

        self.enc_micro = nn.Sequential(
            nn.LayerNorm(16),
            nn.Linear(16, 32),
            nn.SiLU()
        )

        self.state_gru = nn.GRU(
            input_size=208,
            hidden_size=self.state_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.delta_proj = nn.Linear(input_dim, 32, bias=False)
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=0.01)

        self.delta_gru = nn.GRU(
            input_size=32,
            hidden_size=self.delta_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        fused_dim = self.state_hidden_dim + self.delta_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def _compute_micro_features(self, p_b0, v_b0, p_a0, v_a0, p_b1, v_b1, p_a1, v_a1, eps=1e-6):
        def _extract_asset_micro(p_b, v_b, p_a, v_a):
            v_b_pos = v_b.abs() + eps
            v_a_pos = v_a.abs() + eps

            best_bid_p, b_idx = p_b.max(dim=-1, keepdim=True)
            best_ask_p, a_idx = p_a.min(dim=-1, keepdim=True)
            best_bid_v = torch.gather(v_b_pos, dim=-1, index=b_idx)
            best_ask_v = torch.gather(v_a_pos, dim=-1, index=a_idx)

            sum_v_l1 = best_bid_v + best_ask_v + eps
            wap_l1 = (best_bid_p * best_ask_v + best_ask_p * best_bid_v) / sum_v_l1
            imb_l1 = (best_bid_v - best_ask_v) / sum_v_l1
            spread_l1 = best_ask_p - best_bid_p

            total_cash = (p_b * v_b_pos).sum(dim=-1, keepdim=True) + (p_a * v_a_pos).sum(dim=-1, keepdim=True)
            total_vol = v_b_pos.sum(dim=-1, keepdim=True) + v_a_pos.sum(dim=-1, keepdim=True) + eps
            wap_all = total_cash / total_vol
            imb_all = (v_b_pos.sum(dim=-1, keepdim=True) - v_a_pos.sum(dim=-1, keepdim=True)) / total_vol
            wap_diff = wap_all - wap_l1

            return wap_l1, wap_all, wap_diff, spread_l1, imb_l1, imb_all

        w_l1_0, w_all_0, w_diff_0, sp_0, imb_l1_0, imb_all_0 = _extract_asset_micro(p_b0, v_b0, p_a0, v_a0)
        w_l1_1, w_all_1, w_diff_1, sp_1, imb_l1_1, imb_all_1 = _extract_asset_micro(p_b1, v_b1, p_a1, v_a1)

        basis_l1 = w_l1_0 - w_l1_1
        basis_all = w_all_0 - w_all_1
        basis_imb_l1 = imb_l1_0 - imb_l1_1
        basis_imb_all = imb_all_0 - imb_all_1

        return torch.cat([
            w_l1_0, w_all_0, w_diff_0, sp_0, imb_l1_0, imb_all_0,
            w_l1_1, w_all_1, w_diff_1, sp_1, imb_l1_1, imb_all_1,
            basis_l1, basis_all, basis_imb_l1, basis_imb_all
        ], dim=-1)

    def forward(self, x, h=None):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        if h is not None:
            h_state = h[:, :, :self.state_hidden_dim].contiguous()
            h_delta = h[:, :, self.state_hidden_dim:].contiguous()
        else:
            h_state, h_delta = None, None

        p_b0, p_a0 = x_flat[:, 0:11], x_flat[:, 11:22]
        v_b0, v_a0 = x_flat[:, 22:33], x_flat[:, 33:44]
        p_b1, p_a1 = x_flat[:, 52:63], x_flat[:, 63:74]
        v_b1, v_a1 = x_flat[:, 74:85], x_flat[:, 85:96]

        e_p0_lob = self.enc_p0_lob(x_flat[:, 0:22])
        e_v0_lob = self.enc_v0_lob(x_flat[:, 22:44])
        e_p0_ext = self.enc_p0_ext(x_flat[:, 44:48])
        e_v0_ext = self.enc_v0_ext(x_flat[:, 48:52])

        e_p1_lob = self.enc_p1_lob(x_flat[:, 52:74])
        e_v1_lob = self.enc_v1_lob(x_flat[:, 74:96])
        e_p1_ext = self.enc_p1_ext(x_flat[:, 96:100])
        e_v1_ext = self.enc_v1_ext(x_flat[:, 100:104])

        e_aux = self.enc_aux(x_flat[:, 104:112])

        micro_raw = self._compute_micro_features(p_b0, v_b0, p_a0, v_a0, p_b1, v_b1, p_a1, v_a1)
        e_micro = self.enc_micro(micro_raw)

        combined_state = torch.cat([
            e_p0_lob, e_v0_lob, e_p0_ext, e_v0_ext,
            e_p1_lob, e_v1_lob, e_p1_ext, e_v1_ext,
            e_aux, e_micro
        ], dim=-1).view(B, T, 208).to(x.dtype)

        out_state, h_state_next = self.state_gru(combined_state, h_state)

        d_proj = self.delta_proj(x_flat).view(B, T, 32)
        d_diff = torch.cat([
            torch.zeros_like(d_proj[:, :1, :]),
            d_proj[:, 1:, :] - d_proj[:, :-1, :]
        ], dim=1)
        e_delta = torch.tanh(d_diff).to(x.dtype)

        out_delta, h_delta_next = self.delta_gru(e_delta, h_delta)

        fused = torch.cat([out_state, out_delta], dim=-1)
        pred = 2.0 * torch.tanh(self.head(fused))

        h_next = torch.cat([h_state_next, h_delta_next], dim=-1)

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUDualStreamMicroDelta(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )