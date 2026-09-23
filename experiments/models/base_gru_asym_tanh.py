import torch
import torch.nn as nn

class BaselineGRU(nn.Module):
    def __init__(self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.head = nn.Linear(hidden_dim, output_dim)

        self.mu = nn.Parameter(torch.tensor([-0.27, -0.27], dtype=torch.float32))

    def forward(self, x, h=None):
        out, h_next = self.gru(x, h)
        z = self.head(out)

        scale = torch.where(z >= 0, 2.0 - self.mu, 2.0 + self.mu)
        pred = self.mu + scale * torch.tanh(z)

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return BaselineGRU(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim
    )