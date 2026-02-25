#!/usr/bin/env bash
set -euo pipefail

cd /home/zhuonan/code/MedToken

export nnUNet_raw="/home/zhuonan/code/MedToken/data_cache/nnUNet_raw"
export nnUNet_preprocessed="/home/zhuonan/code/MedToken/data_cache/nnUNet_preprocessed"
export nnUNet_results="/home/zhuonan/code/MedToken/outputs/nnunet_results"

DATASET_ID=302
DATASET_NAME="ISIC"
CONFIGURATION="2d"
FOLD=0
TRAINER="nnUNetTrainer"
PLANS="nnUNetPlans"

PYTHONPATH=. python3 tools/prepare_nnunetv2_dataset.py \
  --dataset isic \
  --root data_source/ISIC-2017 \
  --nnunet-raw "${nnUNet_raw}" \
  --dataset-id "${DATASET_ID}" \
  --dataset-name "${DATASET_NAME}" \
  --overwrite

nnUNetv2_plan_and_preprocess \
  -d "${DATASET_ID}" \
  -c "${CONFIGURATION}" \
  --verify_dataset_integrity

nnUNetv2_train \
  "${DATASET_ID}" \
  "${CONFIGURATION}" \
  "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLANS}"

nnUNetv2_predict \
  -d "${DATASET_ID}" \
  -i "${nnUNet_raw}/Dataset${DATASET_ID}_${DATASET_NAME}/imagesTs" \
  -o "outputs/predictions/nnunet_isic_fold${FOLD}" \
  -f "${FOLD}" \
  -c "${CONFIGURATION}" \
  -tr "${TRAINER}" \
  -p "${PLANS}" \
  --save_probabilities
