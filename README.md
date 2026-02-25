# MedToken

## Official nnUNet v2 (TN3K)

`run_nnunet_TN3K.sh` now follows the official nnUNet v2 pipeline:

1. convert TN3K into `nnUNet_raw/DatasetXXX_*` layout
2. run `nnUNetv2_plan_and_preprocess`
3. run `nnUNetv2_train`
4. run `nnUNetv2_predict` on `imagesTs`

The conversion utility is `tools/prepare_nnunetv2_dataset.py` and can also be used for `kvasir` and `isic`.

You can also use official nnUNet v2 architecture directly in MedToken training/inference via:

```bash
python train.py --model nnunet ...
python predict.py --model nnunet --checkpoint <ckpt> ...
```

This model option lazy-loads official nnUNet modules and requires `nnunetv2` to be installed.

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

Expected layout (recommended): `data_source/ISIC-2017/ISIC-2017_Training_Data`, `data_source/ISIC-2017/ISIC-2017_Training_Part1_GroundTruth`, `data_source/ISIC-2017/ISIC-2017_Validation_Data`, and `data_source/ISIC-2017/ISIC-2017_Validation_Part1_GroundTruth`.

Fixed train/val split is used automatically for ISIC-2017. Legacy layouts are still supported: `images/*` + `masks/*`, or `img/*` + `label/*`.

```bash
python predict.py \
	--checkpoint outputs/logs/<run>/checkpoints/<ckpt>.ckpt \
	--isic-root data_source/ISIC-2017 \
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
