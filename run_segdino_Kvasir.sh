cd /home/zhuonan/code/MedToken
python train.py \
  --model segdino \
  --run-model Segdino-kvasir \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --kvasir-root data_source/Kvasir \
  --image-size 256 \
  --batch-size 4 \
  --max-epochs 50 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --loss dc_bce \
  --weight-dice 0.0 \
  --segdino-encoder-size small \
  --seg-preprocess dino \
  --lr-scheduler cosine \
  --cosine-t-max 50
