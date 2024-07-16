wandb_path="../"
save_path="../"
export PYTHONPATH="."

# base_tasks
sbatch submit.slurm python mask_train.py \
	--subtask="base_tasks" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \


# remove_second
sbatch submit.slurm  python mask_train.py \
	--subtask="remove_second" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \


# remove_first
 sbatch submit.slurm python mask_train.py \
	--subtask="remove_first" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \


# copy
 sbatch submit.slurm python mask_train.py \
	--subtask="copy" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=200 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \


# append
 sbatch submit.slurm python mask_train.py \
	--subtask="append" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=300 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-8 \
	--lr=1e-2 \

# echo
 sbatch submit.slurm python mask_train.py \
	--subtask="echo" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \


# prepend
 sbatch submit.slurm python mask_train.py \
	--subtask="prepend" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \

# shift
 sbatch submit.slurm python mask_train.py \
	--subtask="shift" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \

# swap_first_last
 sbatch submit.slurm python mask_train.py \
	--subtask="swap_first_last" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \


# reverse
 sbatch submit.slurm python mask_train.py \
	--subtask="reverse" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-7 \
	--lr=1e-2 \

# repeat
 sbatch submit.slurm python mask_train.py \
	--subtask="repeat" \
	--save_path=$save_path \
	--wandb_path=$wandb_path \
	--epochs=100 \
	--pruning_method="continuous" \
	--max_temp=100 \
	--mask_initial_value=0.1\
	--mask_lambda=1e-8 \
	--lr=1e-2 \
