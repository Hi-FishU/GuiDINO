# MedToken

## Inference (nnUNet-style)

Run sliding-window inference with Gaussian blending and optional mirroring/TTA using [predict.py](predict.py).

### Kvasir

```bash
python predict.py \
	--checkpoint outputs/logs/<run>/checkpoints/<ckpt>.ckpt \
	--kvasir-root data_source/Kvasir \
	--model nnwnet \
	--patch-size 256 256 \
	--overlap 0.5 \
	--mirror-tta
```

### DRIVE (test split)

```bash
python predict.py \
	--checkpoint outputs/logs/<run>/checkpoints/<ckpt>.ckpt \
	--drive-root data_source/drive \
	--drive-split test \
	--model unet \
	--patch-size 512 512
```

### Synapse (2D slices)

```bash
python predict.py \
	--checkpoint outputs/logs/<run>/checkpoints/<ckpt>.ckpt \
	--synapse-root data_source/synapse \
	--model nnwnet \
	--synapse-target-spacing 1.0 1.0 1.0 \
	--patch-size 256 256
```