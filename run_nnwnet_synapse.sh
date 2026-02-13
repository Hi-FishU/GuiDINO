cd /home/zhuonan/code/MedToken
python train.py \
  --model nnwnet \
  --run-model nnwnet-synapse \
  --synapse-root data_source/synapse \
  --loss dc_ce \
  --image-size 512 \
  --batch-size 4 \
  --max-epochs 2000 \
  --lr 1e-2 \
  --weight-decay 3e-5 \
  --optimizer sgd \
  --num-classes 9 \
  --in-chans 1 \
  --no-train-epoch-eval \
  --no-amp
