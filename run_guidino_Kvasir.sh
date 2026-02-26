cd /home/zhuonan/code/MedToken
python train.py \
  --use-guide \
  --run-model guidino-kvasir \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --loss guide_dc_bce \
  --image-size 512 \
  --batch-size 4 \
  --max-epochs 400 \
  --seg-preprocess dino_strong \
  --optimizer sgd \
  --lr 1e-2 \
  --weight-decay 3e-5 \
  --dice-do-bg \
  --weight-guide 0.1 \
  --tokenbook-sample-rate 1.0 \
  --tokenbook-dropout 0.0 \
  --nnwnet-deep-supervision \
  --no-train-epoch-eval \
