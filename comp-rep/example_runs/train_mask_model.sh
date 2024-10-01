save_path="../model_checkpoints"
wandb_path="../"
cache_dir="../../data/cached_probabilities"  # Path to the cached probabilities of the base model
export PYTHONPATH="../"

python ../mask_train.py \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--cache_dir=$cache_dir \
	--epochs=500 \
	--eval \
	--pruning_method="continuous" \
	--faithfulness_freq=20 \
	--subtask="copy" \
	--max_temp=200 \
	--mask_lambda=1e-4 \
	--mask_initial_value=0.05 \
	--lr=1e-4 \
	--ablation_value="mean"
