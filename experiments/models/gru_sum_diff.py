import torch
import torch.nn as nn


class BaselineGRUSumDiff(nn.Module):
    def __init__(self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.head_diff = nn.Linear(hidden_dim, 1)
        self.head_sum = nn.Linear(hidden_dim, 1)

    def forward(self, x, h=None):
        out, h_next = self.gru(x, h)

        z_diff = self.head_diff(out)
        z_sum = self.head_sum(out)

        p0 = 0.5 * (z_sum + z_diff)
        p1 = 0.5 * (z_sum - z_diff)

        pred = 2.0 * torch.tanh(torch.cat([p0, p1], dim=-1))
        return pred, h_next


def create_model(cfg) -> nn.Module:
    return BaselineGRUSumDiff(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim
    )