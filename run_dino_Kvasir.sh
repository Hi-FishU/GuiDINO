cd /home/zhuonan/code/MedToken
python train.py \
  --model dinov3 \
  --run-model dino-kvasir \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --kvasir-root data_source/Kvasir \
  --loss dc_bce \
  --max-epochs 2000 \
  --image-size 256 \
  --seg-preprocess dino_strong \
