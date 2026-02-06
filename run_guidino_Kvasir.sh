cd /home/zhuonan/code/MedToken
python train.py \
  --use-guide \
  --run-model guidino-kvasir \
  --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
  --kvasir-root data_source/Kvasir \
  --max-epochs 2000 \
  --image-size 512 \
  --seg-preprocess dino \
  --weight-guide 0.1 \
  --lr-scheduler cosine \
  --tokenbook-dropout 0.5 \
  --tokenbook-sample-rate 0.5 \
