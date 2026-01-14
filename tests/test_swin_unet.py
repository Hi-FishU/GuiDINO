import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_swin_unet_module():
    module_path = Path(__file__).resolve().parents[1] / "model" / "swin-unet.py"
    spec = importlib.util.spec_from_file_location("swin_unet_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_swin_unet_forward_shape():
    module = _load_swin_unet_module()
    model = module.SwinUnet(img_size=224, patch_size=4, in_chans=3, num_classes=1)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 224, 224)
    assert torch.isfinite(out).all()


def test_swin_unet_size_mismatch_raises():
    module = _load_swin_unet_module()
    model = module.SwinUnet(img_size=224, patch_size=4, in_chans=3, num_classes=1)
    x = torch.randn(1, 3, 32, 224)
    with pytest.raises(ValueError, match="Input image size"):
        model(x)
