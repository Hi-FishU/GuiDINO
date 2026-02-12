cd /home/zhuonan/code/MedToken
python train.py \
    --model dinov3 \
    --use-guide \
    --run-model guidino-synapse \
    --dinov3-backbone facebook/dinov3-vits16-pretrain-lvd1689m \
    --synapse-root data_source/synapse \
    --loss guide_dc_ce \
    --max-epochs 2000 \
    --num-classes 9 \
    --in-chans 1 \
    --image-size 512 \
    --optimizer sgd \
    --lr-scheduler cosine_restart \
    --seg-preprocess dino_strong \
    --tokenbook-dropout 0.8 \
    --tokenbook-sample-rate 0.5 \
    --tokenbook-use-ema \
