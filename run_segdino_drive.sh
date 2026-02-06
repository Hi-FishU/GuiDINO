cd /home/zhuonan/code/MedToken
python train.py \
  --model segdino \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --drive-root data_source/drive \
  --image-size 256 \
  --batch-size 4 \
  --max-epochs 50 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --optimizer adamw \
  --loss dc_bce \
  --weight-dice 0.0 \
  --segdino-encoder-size small \
  --seg-preprocess dino
