import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


class TokenBook(nn.Module):
    def __init__(
        self,
        n_tokens,
        embed_dim,
        height=None,
        width=None,
        dropout: float = 0.0,
        ema_decay: float | None = None,
        use_ema: bool = False,
    ):
        super().__init__()
        self.input_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.book = nn.Parameter(torch.randn(n_tokens, embed_dim))
        nn.init.normal_(self.book, 0, embed_dim ** -0.5)
        self.ema_decay = ema_decay
        self.use_ema = bool(use_ema)
        if ema_decay is not None:
            self.register_buffer("book_ema", self.book.detach().clone())
        else:
            self.book_ema = None
        # note this is the latent image's height and width
        self.height, self.width = height, width
        self.dropout = float(dropout)

    @torch.no_grad()
    def _ema_update(self) -> None:
        if self.ema_decay is None or self.book_ema is None:
            return
        decay = float(self.ema_decay)
        self.book_ema.mul_(decay).add_(self.book.detach(), alpha=1.0 - decay)

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
        if self.training:
            self._ema_update()
        B = x.shape[0]
        book = self.book_ema if self.use_ema and self.book_ema is not None else self.book
        sims = book.unsqueeze(0).expand(B, -1, -1)
        x = self.input_conv(x.transpose(1, 2).reshape(B, -1, height, width)).flatten(2).transpose(1, 2)
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
