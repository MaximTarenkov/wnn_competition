import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUChampionPure(nn.Module):
    def __init__(
        self, input_dim=112, hidden_dim=128, num_layers=2, output_dim=2
    ):
        super().__init__()

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

        # 1 x Быстрый BBO-энкодер (вместо медленного max/gather берет линейные комбинации колонок)
        self.enc_l1_fast = nn.Sequential(nn.Linear(44, 24), nn.SiLU())

        # =====================================================================
        # РЕЗКИЙ ИМПУЛЬСНЫЙ БЛОК (k=1 и k=2): 112 -> 32
        # =====================================================================
        self.delta_proj = nn.Linear(input_dim, 32, bias=False)

        # 32*4 (128) + 8*4 (32) + 16 (aux) + 24 (l1) + 32 (delta1) + 16 (delta2) = 248
        self.delta2_proj = nn.Linear(32, 16, bias=False)

        self.gru = nn.GRU(
            input_size=248,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # 1. СТАКАН: Чистые 2D GEMM проекции (без max/min/gather!)
        e_p0 = self.enc_p0_lob(x_flat[:, 0:22])
        e_v0 = self.enc_v0_lob(x_flat[:, 22:44])
        e_p0_ext = self.enc_p0_ext(x_flat[:, 44:48])
        e_v0_ext = self.enc_v0_ext(x_flat[:, 48:52])

        e_p1 = self.enc_p1_lob(x_flat[:, 52:74])
        e_v1 = self.enc_v1_lob(x_flat[:, 74:96])
        e_p1_ext = self.enc_p1_ext(x_flat[:, 96:100])
        e_v1_ext = self.enc_v1_ext(x_flat[:, 100:104])

        e_aux = self.enc_aux(x_flat[:, 104:112])
        
        # Быстрый L1 без gather: все 44 цены двух активов залетают в один слой
        e_l1 = self.enc_l1_fast(torch.cat([x_flat[:, 0:22], x_flat[:, 52:74]], dim=-1))

        # 2. РЕЗКИЙ ИМПУЛЬС (1 и 2 тика):
        d_proj = self.delta_proj(x_flat).view(B, T, 32)
        
        # Дельта 1 тик (микро-удар)
        d1 = torch.cat([torch.zeros_like(d_proj[:, :1, :]), d_proj[:, 1:, :] - d_proj[:, :-1, :]], dim=1)
        e_d1 = F.silu(d1).reshape(B * T, 32)

        # Дельта 2 тика (ускорение)
        d2 = torch.cat([torch.zeros_like(d_proj[:, :2, :]), d_proj[:, 2:, :] - d_proj[:, :-2, :]], dim=1)
        e_d2 = F.silu(self.delta2_proj(d2.reshape(B * T, 32)))

        # 3. СБОРКА И cuDNN GRU (dim = 248)
        combined = torch.cat([
            e_p0, e_v0, e_p0_ext, e_v0_ext,
            e_p1, e_v1, e_p1_ext, e_v1_ext,
            e_aux, e_l1, e_d1, e_d2
        ], dim=-1).view(B, T, 248).to(x.dtype)

        out, h_next = self.gru(combined, h)
        pred = 2.0 * torch.tanh(self.head(out))

        return pred, h_next


def create_model(cfg) -> nn.Module:
    return GRUChampionPure(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=cfg.output_dim,
    )