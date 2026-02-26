cd /home/zhuonan/code/MedToken
ISIC_ROOT="${ISIC_ROOT:-data_source/ISIC-2017}"
NUM_WORKERS="${NUM_WORKERS:-12}"
python train.py \
  --model guidedino \
  --run-model guidino-isic \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --isic-root "$ISIC_ROOT" \
  --loss guide_dc_bce \
  --image-size 512 \
  --batch-size 4 \
  --num-workers "$NUM_WORKERS" \
  --dataloader-mp-context fork \
  --max-epochs 400 \
  --seg-preprocess dino_strong \
  --optimizer sgd \
  --lr 1e-2 \
  --lr-scheduler cosine \
  --weight-decay 3e-5 \
  --dice-do-bg \
  --weight-guide 0.1 \
  --tokenbook-sample-rate 1.0 \
  --tokenbook-dropout 0.0 \
  --nnwnet-deep-supervision \
  --no-fullres-val-eval \
  --check-val-every-n-epoch 5 \
  --limit-val-batches 0.25 \
  --num-sanity-val-steps 0 \
  --no-train-epoch-eval \
