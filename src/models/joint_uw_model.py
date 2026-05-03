# scr/model.py

import torch
import torch.nn as nn


class ProbabilisticSingleFrameYtoU(nn.Module):
    """
    Probabilistic model for:

        sparse 2D displacement at one frame
        y_t: [B, K*2]

        -> full-field 3D displacement
        U_t: [B, N*3]

    Output:
        U_mu     : predicted mean displacement
        U_logvar : predicted log variance
        z_mu     : latent mean
        z_logvar : latent log variance
    """

    def __init__(
        self,
        y_dim: int,
        U_dim: int,
        z_dim: int = 32,
        hidden_dim: int = 256,
        logvar_min: float = -20.0,
        logvar_max: float = -10.0,
    ):
        super().__init__()

        self.y_dim = y_dim
        self.U_dim = U_dim
        self.z_dim = z_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.encoder = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.z_mu = nn.Linear(hidden_dim // 2, z_dim)
        self.z_logvar = nn.Linear(hidden_dim // 2, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, U_dim * 2),
        )

    def encode(self, y):
        h = self.encoder(y)
        z_mu = self.z_mu(h)
        z_logvar = self.z_logvar(h)
        return z_mu, z_logvar

    def reparameterize(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        z = z_mu + eps * z_std
        return z

    def bound_logvar(self, raw_logvar):
        """
        Soft bound log variance into [logvar_min, logvar_max].
        """
        return self.logvar_min + (self.logvar_max - self.logvar_min) * torch.sigmoid(raw_logvar)

    def decode(self, z):
        out = self.decoder(z)
        U_mu, raw_U_logvar = torch.chunk(out, 2, dim=1)
        U_logvar = self.bound_logvar(raw_U_logvar)
        return U_mu, U_logvar

    def forward(self, y, sample_z=True):
        z_mu, z_logvar = self.encode(y)

        if sample_z:
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z = z_mu

        U_mu, U_logvar = self.decode(z)
        return U_mu, U_logvar, z_mu, z_logvar


def kl_loss(z_mu, z_logvar):
    """
    KL divergence between q(z|y) and N(0, I).
    """
    return -0.5 * torch.mean(
        1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp()
    )


def gaussian_nll_loss(U, U_mu, U_logvar):
    """
    Gaussian negative log likelihood.

    U ~ N(U_mu, exp(U_logvar))
    """
    return 0.5 * torch.mean(
        torch.exp(-U_logvar) * (U - U_mu) ** 2 + U_logvar
    )


def mse_loss(U, U_mu):
    return nn.functional.mse_loss(U_mu, U)