import torch
import torch.nn as nn


class JointUWProbabilisticMLP(nn.Module):
    def __init__(
        self,
        y_dim: int,
        U_dim: int,
        K_obs: int,
        N_vertices: int,
        z_dim: int = 32,
        hidden_dim: int = 256,
        logvar_min: float = -8.0,
        logvar_max: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.U_dim = U_dim
        self.K_obs = K_obs
        self.N_vertices = N_vertices
        self.z_dim = z_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.encoder = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.z_mu_head = nn.Linear(hidden_dim, z_dim)
        self.z_logvar_head = nn.Linear(hidden_dim, z_dim)

        self.decoder_backbone = nn.Sequential(
            nn.Linear(y_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.U_mu_head = nn.Linear(hidden_dim, U_dim)
        self.U_logvar_head = nn.Linear(hidden_dim, U_dim)
        self.W_head = nn.Linear(hidden_dim, K_obs * N_vertices)

    def reparameterize(self, mu, logvar, sample_z=True):
        if not sample_z:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, y_norm, sample_z=True):
        h = self.encoder(y_norm)
        z_mu = self.z_mu_head(h)
        z_logvar = self.z_logvar_head(h).clamp(min=-12.0, max=8.0)

        z = self.reparameterize(z_mu, z_logvar, sample_z=sample_z)
        hd = self.decoder_backbone(torch.cat([y_norm, z], dim=-1))

        U_mu = self.U_mu_head(hd)
        U_logvar = self.U_logvar_head(hd).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        W_logits = self.W_head(hd).view(-1, self.K_obs, self.N_vertices)

        return U_mu, U_logvar, W_logits, z_mu, z_logvar
