import torch
import torch.nn.functional as F


def mse_loss(target, pred):
    return torch.mean((target - pred) ** 2)


def gaussian_nll_loss(target, mu, logvar):
    return 0.5 * torch.mean(
        logvar + (target - mu).pow(2) / torch.exp(logvar)
    )


def kl_loss(z_mu, z_logvar):
    return -0.5 * torch.mean(
        1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp()
    )


def soft_ce_loss_from_sparse(logits, sparse_idx, sparse_val):
    """
    logits:     [B, K, N]
    sparse_idx: [B, K, 3]
    sparse_val: [B, K, 3]
    """
    logp = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(logp, dim=-1, index=sparse_idx.long())
    ce = -(sparse_val * gathered).sum(dim=-1)
    return ce.mean()
