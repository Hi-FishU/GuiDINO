# cd /home/zhuonan/code/MedToken
ISIC_ROOT="${ISIC_ROOT:-data_cache/ISIC_1024}"
python train.py \
  --model guidennwnet \
  --run-model guidennwnet-ISIC \
  --isic-root "$ISIC_ROOT" \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --loss guide_dc_bce \
  --image-size 512 \
  --batch-size 4 \
  --max-epochs 400 \
  --seg-preprocess dino_strong \
  --optimizer adamw \
  --lr 1e-2 \
  --lr-scheduler cosine_restart \
  --weight-decay 3e-5 \
  --dice-do-bg \
  --weight-guide 0.1 \
  --tokenbook-sample-rate 1.0 \
  --tokenbook-dropout 0.0 \
  --nnwnet-deep-supervision \
  --no-train-epoch-eval
