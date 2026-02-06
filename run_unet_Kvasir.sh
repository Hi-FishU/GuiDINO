cd /home/zhuonan/code/MedToken
python train.py \
  --model unet \
  --run-model unet-kvasir \
  --kvasir-root data_source/Kvasir \
  --loss dc_bce \
  --image-size 512 \
  --no-train-epoch-eval
