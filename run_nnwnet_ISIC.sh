cd /home/zhuonan/code/MedToken
ISIC_ROOT="${ISIC_ROOT:-data_cache/ISIC_1024}"
python train.py \
  --model nnwnet \
  --run-model nnwnet-isic \
  --isic-root "$ISIC_ROOT" \
  --loss dc_bce \
  --image-size 512 \
  --batch-size 4 \
  --max-epochs 400 \
  --num-workers 8 \
  --dataloader-mp-context fork \
  --lr 1e-2 \
  --weight-decay 3e-5 \
  --optimizer sgd \
  --no-train-epoch-eval \
  --no-amp
