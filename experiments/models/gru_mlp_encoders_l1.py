import torch
import torch.nn as nn

class GRUWithEncoders(nn.Module):

    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        self.price_indices = (
            list(range(0, 22))
            + list(range(44, 48))
            + list(range(52, 74))
            + list(range(96, 100))
        )
        self.vol_indices = (
            list(range(22, 44))
            + list(range(48, 52))
            + list(range(74, 96))
            + list(range(100, 104))
        )
        self.add_indices = list(range(104, 112))

        self.price_encoder = nn.Sequential(
            nn.Linear(len(self.price_indices), 64), nn.SiLU()
        )
        self.vol_encoder = nn.Sequential(
            nn.Linear(len(self.vol_indices), 64), nn.SiLU()
        )
        self.add_encoder = nn.Sequential(
            nn.Linear(len(self.add_indices), 32), nn.SiLU()
        )

        self.l1_encoder = nn.Sequential(nn.Linear(5, 16), nn.SiLU())

        self.gru = nn.GRU(
            input_size=176,  # 64 + 64 + 32 + 16
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def _extract_l1(self, p_b, v_b, p_a, v_a):
        mask_b = (v_b > 0.0) & (p_b > 0.0)
        p_b_key = torch.where(mask_b, p_b, torch.full_like(p_b, -1e6)).detach()
        bid_idx = torch.argmax(p_b_key, dim=-1, keepdim=True)

        best_bid_p = torch.gather(p_b, dim=-1, index=bid_idx)
        best_bid_v = torch.gather(v_b, dim=-1, index=bid_idx)

        has_valid_b = mask_b.any(dim=-1, keepdim=True)
        best_bid_p = torch.where(has_valid_b, best_bid_p, torch.zeros_like(best_bid_p))
        best_bid_v = torch.where(has_valid_b, best_bid_v, torch.zeros_like(best_bid_v))

        mask_a = (v_a > 0.0) & (p_a > 0.0)
        p_a_key = torch.where(mask_a, p_a, torch.full_like(p_a, 1e6)).detach()
        ask_idx = torch.argmin(p_a_key, dim=-1, keepdim=True)

        best_ask_p = torch.gather(p_a, dim=-1, index=ask_idx)
        best_ask_v = torch.gather(v_a, dim=-1, index=ask_idx)

        has_valid_a = mask_a.any(dim=-1, keepdim=True)
        best_ask_p = torch.where(has_valid_a, best_ask_p, best_bid_p)
        best_ask_v = torch.where(has_valid_a, best_ask_v, torch.zeros_like(best_ask_v))

        spread = torch.clamp(best_ask_p - best_bid_p, min=0.0)
        mid = 0.5 * (best_bid_p + best_ask_p)

        vol_sum = best_bid_v + best_ask_v + 1e-6
        imbalance = torch.clamp((best_bid_v - best_ask_v) / vol_sum, -1.0, 1.0)

        return spread, imbalance, mid

    def forward(self, x, h=None):
        p = self.price_encoder(x[:, :, self.price_indices])
        v = self.vol_encoder(x[:, :, self.vol_indices])
        a = self.add_encoder(x[:, :, self.add_indices])

        s0, imb0, mid0 = self._extract_l1(
            x[:, :, 0:11], x[:, :, 22:33], x[:, :, 11:22], x[:, :, 33:44]
        )
        s1, imb1, mid1 = self._extract_l1(
            x[:, :, 52:63], x[:, :, 74:85], x[:, :, 63:74], x[:, :, 85:96]
        )
        mid_diff = mid0 - mid1

        l1_raw = torch.cat([s0, imb0, s1, imb1, mid_diff], dim=-1)
        l1 = self.l1_encoder(l1_raw)

        combined = torch.cat([p, v, a, l1], dim=-1)

        out, h_next = self.gru(combined, h)

        pred = 2.0 * torch.tanh(self.head(out))

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUWithEncoders(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )

