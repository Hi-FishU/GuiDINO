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

### ISIC

Expected layout: either `data_source/ISIC/images/*` + `data_source/ISIC/masks/*`, or `data_source/ISIC/img/*` + `data_source/ISIC/label/*` with matching filenames.

```bash
python predict.py \
	--checkpoint outputs/logs/<run>/checkpoints/<ckpt>.ckpt \
	--isic-root data_source/ISIC \
	--model guidedino \
	--patch-size 512 512 \
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

## Offline Evaluation (Dice/IoU + ASD/HD95)

Use `evaluate_offline.py` to evaluate saved prediction masks against GT with optional nnUNet-style postprocessing selection.

```bash
python evaluate_offline.py \
	--pred-dir data_source/Kvasir/images \
	--kvasir-root data_source/Kvasir \
	--auto-postprocess-largest \
	--surface-empty-policy penalize \
	--surface-aggregation nonempty \
	--output-json outputs/kvasir_summary_boundary.json
```
