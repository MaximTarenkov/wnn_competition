import torch
import torch.nn as nn


class GRUWithEncoders(nn.Module):

    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

        self.price_encoder = nn.Sequential(nn.Linear(52, 64), nn.SiLU())
        self.vol_encoder = nn.Sequential(nn.Linear(52, 64), nn.SiLU())
        self.add_encoder = nn.Sequential(nn.Linear(8, 32), nn.SiLU())

        self.gru = nn.GRU(
            input_size=160,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def _process_book(self, p_b, v_b, p_a, v_a, dp, dv):
        p_b_safe = torch.where(
            v_b == 0.0, torch.full_like(p_b, -float("inf")), p_b
        )
        sorted_p_b, idx_b = torch.sort(p_b_safe, dim=-1, descending=True)
        sorted_v_b = torch.gather(v_b, dim=-1, index=idx_b)
        sorted_p_b = torch.where(
            torch.isinf(sorted_p_b), torch.zeros_like(sorted_p_b), sorted_p_b
        )

        p_a_safe = torch.where(
            v_a == 0.0, torch.full_like(p_a, float("inf")), p_a
        )
        sorted_p_a, idx_a = torch.sort(p_a_safe, dim=-1, descending=False)
        sorted_v_a = torch.gather(v_a, dim=-1, index=idx_a)
        sorted_p_a = torch.where(
            torch.isinf(sorted_p_a), torch.zeros_like(sorted_p_a), sorted_p_a
        )

        p_feats = torch.cat([sorted_p_b, sorted_p_a, dp], dim=-1)
        v_feats = torch.cat([sorted_v_b, sorted_v_a, dv], dim=-1)

        return p_feats, v_feats

    def forward(self, x, h=None):
        p0, v0 = self._process_book(
            x[:, :, 0:11],
            x[:, :, 22:33],
            x[:, :, 11:22],
            x[:, :, 33:44],
            x[:, :, 44:48],
            x[:, :, 48:52],
        )
        p1, v1 = self._process_book(
            x[:, :, 52:63],
            x[:, :, 74:85],
            x[:, :, 63:74],
            x[:, :, 85:96],
            x[:, :, 96:100],
            x[:, :, 100:104],
        )

        p_all = torch.cat([p0, p1], dim=-1)
        v_all = torch.cat([v0, v1], dim=-1)
        a_all = x[:, :, 104:112]

        p = self.price_encoder(p_all)
        v = self.vol_encoder(v_all)
        a = self.add_encoder(a_all)

        combined = torch.cat([p, v, a], dim=-1)

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

