cd /home/zhuonan/code/MedToken
python train.py \
  --model guideunet \
  --run-model guideunet-kvasir \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --kvasir-root data_source/Kvasir \
  --loss guide_dc_bce \
  --image-size 512 \
  --seg-preprocess dino \
  --weight-guide 0.1 \
  --tokenbook-dropout 0.5 \
  --tokenbook-sample-rate 0.5 \
  --log-image-samples 2 \
  --log-image-every-n-epochs 1 \
  --no-train-epoch-eval \
  --lr-scheduler cosine_restart \
