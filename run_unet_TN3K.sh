cd /home/zhuonan/code/MedToken
python train.py \
  --model unet \
  --run-model unet-tn3k \
  --tn3k-root data_source/Thyoid/tn3k \
  --tn3k-use-test-as-val \
  --loss dc_bce \
  --image-size 352 \
  --batch-size 4 \
  --max-epochs 400 \
  --seg-preprocess nnunet \
  --optimizer adamw \
  --lr 1e-4 \
  --lr-scheduler poly \
  --weight-decay 1e-4 \
  --dice-do-bg \
  --weight-guide 0.1 \
  --no-train-epoch-eval