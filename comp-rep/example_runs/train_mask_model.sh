save_path="../model_checkpoints"
wandb_path="../"


python ../comp_rep/mask_train.py \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=20 \
	--eval \
	--pruning_method="continuous" \
	--max_temp=200 \
	--mask_lambda=1e-7 \
	--lr=1e-4 \
