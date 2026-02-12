cd /home/zhuonan/code/MedToken
python train.py \
	--model nnwnet \
	--run-model nnwnet-kvasir \
	--kvasir-root data_source/Kvasir \
	--loss dc_bce \
	--image-size 512 \
	--batch-size 4 \
	--max-epochs 400 \
	--lr 1e-2 \
	--weight-decay 3e-5 \
	--optimizer sgd \
	--no-train-epoch-eval \
	--no-amp \
