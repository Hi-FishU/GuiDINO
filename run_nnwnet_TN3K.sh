cd /home/zhuonan/code/MedToken
python train.py \
  --model nnwnet \
  --run-model nnwnet-tn3k \
  --tn3k-root data_source/Thyoid/tn3k \
  --loss dc_bce \
  --image-size 512 \
  --batch-size 4 \
  --max-epochs 400 \
  --seg-preprocess nnunet \
  --optimizer sgd \
  --lr 1e-2 \
  --lr-scheduler cosine \
  --weight-decay 3e-5 \
  --dice-do-bg \
  --nnwnet-deep-supervision \
  --no-train-epoch-eval


