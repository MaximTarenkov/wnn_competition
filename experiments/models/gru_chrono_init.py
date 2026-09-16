import numpy as np
import torch
import torch.nn as nn


class BaselineGRUChrono(nn.Module):
    def __init__(
        self,
        input_dim=112,
        hidden_dim=128,
        num_layers=2,
        output_dim=2,
        t_max=150.0,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "bias_ih_l0" in name:
                    h_dim = param.size(0) // 3
                    param[0:h_dim].fill_(2.5)

                    log_t = torch.empty(h_dim).uniform_(0.0, np.log(t_max))
                    t = torch.exp(log_t)
                    param[h_dim : 2 * h_dim].copy_(
                        torch.log(torch.clamp(t - 1.0, min=1e-4))
                    )
                    param[2 * h_dim :].fill_(0.0)
                elif "bias_hh_l0" in name:
                    param.fill_(0.0)

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        out, h_next = self.gru(x, h)
        pred = 2.0 * torch.tanh(self.head(out))
        return pred, h_next


def create_model(cfg) -> nn.Module:
    return BaselineGRUChrono(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )