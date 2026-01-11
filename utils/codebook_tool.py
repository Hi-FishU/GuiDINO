import math
from typing import Iterator
import numpy as np
import torch


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def top_p_mean(a: np.ndarray, p: float = 0.02) -> float:
    """Mean of top p fraction (e.g., top 2% tokens)."""
    if a.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(p * a.size)))
    idx = np.argpartition(a, -k)[-k:]
    return float(a[idx].mean())


def iter_images_from_loader(loader) -> Iterator[torch.Tensor]:
    """
    Expects loader yields either:
      - images
      - (images, labels, ...)
    Returns images as float tensor [B, C, H, W].
    """
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        yield x