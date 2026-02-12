cd /home/zhuonan/code/MedToken
python train.py \
  --use-guide \
  --run-model guidino-kvasir \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --kvasir-root data_source/Kvasir \
  --max-epochs 2000 \
  --image-size 512 \
  --seg-preprocess dino_strong \
  --loss guide_dc_bce_hinged \
  --weight-guide 0.1 \
  --weight-hinge-d 0.05 \
  --hinge-d-margin 1.0 \
  --hinge-d-kernel-size 3 \
  --lr-scheduler cosine_restart \
  --tokenbook-dropout 0.8 \
  --tokenbook-sample-rate 0.5 \
  --tokenbook-use-ema
