save_path="../model_checkpoints"
wandb_path="../"
export PYTHONPATH="../"

python ../train.py \
	--lr 7e-5 \
	--hidden_size 512 \
	--layers 6 \
	--epochs 20 \
	--eval \
	--val_batch_size 64 \
	--eval \
	--save_path=$save_path \
	--wandb_path=$wandb_path
