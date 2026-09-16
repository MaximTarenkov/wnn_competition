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

        self.gru = nn.GRU(
            input_size=160,  # 64 + 64 + 32
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        p = self.price_encoder(x[:, :, self.price_indices])
        v = self.vol_encoder(x[:, :, self.vol_indices])
        a = self.add_encoder(x[:, :, self.add_indices])

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

