cd /home/zhuonan/code/MedToken
python train.py \
--model dinov3 \
--run-model dino-synapse \
--dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
--synapse-root data_source/synapse \
--loss dc_ce \
--max-epochs 2000 \
--num-classes 9 \
--in-chans 1 \
--image-size 512 \
--seg-preprocess dino