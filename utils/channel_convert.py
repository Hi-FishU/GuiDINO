import torch
import torch.nn.functional as F


def sobel_mag(x):  # x: [1,H,W], returns [1,H,W]
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                      dtype=x.dtype, device=x.device).view(1, 1, 3, 3)/8
    kx = ky.transpose(2, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)


def minmax_norm(t, eps=1e-6):
    mn, mx = t.amin(dim=(-2, -1), keepdim=True), t.amax(dim=(-2, -1), keepdim=True)
    return (t - mn) / (mx - mn + eps)


def make_pseudo_rgb(x):
    """x: [H,W], returns [3,H,W]"""
    if isinstance(x, torch.Tensor) is False:
        x = torch.tensor(x)
    x = x.unsqueeze(0)  # [1,H,W]
    raw = minmax_norm(x)
    # simple local contrast: unsharp mask style
    blur = F.avg_pool2d(raw, 5, stride=1, padding=2)
    high = minmax_norm(raw - blur)
    edges = minmax_norm(sobel_mag(raw))
    # [3,H,W]
    return torch.cat([raw, high, edges], dim=0).permute(1, 2, 0).numpy()
