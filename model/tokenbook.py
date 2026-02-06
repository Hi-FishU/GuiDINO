import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


class TokenBook(nn.Module):
    def __init__(self, n_tokens, embed_dim, height=None, width=None, dropout: float = 0.0):
        super().__init__()
        self.book = nn.Parameter(torch.randn(n_tokens, embed_dim))
        nn.init.normal_(self.book, 0, embed_dim ** -0.5)
        # note this is the latent image's height and width
        self.height, self.width = height, width
        self.dropout = float(dropout)

    def forward(self, x, height=None, width=None, token_mask: torch.Tensor | None = None):
        if height is None:
            height = self.height
        if width is None:
            width = self.width
        if height is None or width is None:
            length = x.shape[1]
            side = int(length ** 0.5)
            if side * side != length:
                raise ValueError(
                    f"TokenBook needs height/width; cannot infer from length={length}."
                )
            height = side
            width = side
        B = x.shape[0]
        sims = self.book.unsqueeze(0).expand(B, -1, -1)
        x = F.normalize(x, dim=-1)
        sims = F.normalize(sims, dim=-1)
        sims = einsum(sims, x, 'b n c, b l c -> b l n')
        if self.dropout > 0:
            sims = F.dropout(sims, p=self.dropout, training=self.training)
        sims = torch.mean(sims, dim=-1)  # B, L
        if token_mask is not None:
            token_mask = token_mask.to(sims.dtype)
            sims = sims * token_mask
            denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            sims = sims / denom
        sims = rearrange(sims, 'b (h w) -> b h w',
                         h=height, w=width).unsqueeze(1)
        sims = sims * 0.5 + 0.5  # normalize to [0, 1]
        return sims  # B, 1, H, W
