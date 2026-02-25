#!/usr/bin/env bash
set -euo pipefail

cd /home/zhuonan/code/MedToken

python train.py \
  --model nnunet \
  --run-model nnunet-tn3k \
  --tn3k-root data_source/Thyoid/tn3k \
  --tn3k-use-test-as-val \
  --loss dc_bce \
  --image-size 512 \
  --batch-size 4 \
  --max-epochs 400 \
  --seg-preprocess nnunet \
  --optimizer sgd \
  --lr 1e-2 \
  --lr-scheduler cosine \
  --weight-decay 3e-5 \
  --dice-do-bg \
  --nnunet-stages 6 \
  --nnunet-base-features 32 \
  --nnunet-max-features 512 \
  --nnwnet-deep-supervision \
  --no-train-epoch-eval
