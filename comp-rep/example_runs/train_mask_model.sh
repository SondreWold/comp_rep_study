train_mask_path="../../data/function_tasks/append/train.csv"
val_mask_path="../../data/function_tasks/append/test.csv"
pretrained_model_path="../model_checkpoints/pcfgs_base/model.ckpt"
tokenizer_path="../model_checkpoints/pcfgs_base/"
save_path="../masked_models/append/"
predictions_path="../masked_models/append"


python ../comp_rep/mask_train.py \
	--train_mask_path=$train_mask_path \
	--val_mask_path=$val_mask_path \
	--pretrained_model_path=$pretrained_model_path \
	--tokenizer_path=$tokenizer_path \
	--epochs=20 \
	--eval \
	--save_path=$save_path \
	--predictions_path=$predictions_path \
	--pruning_method="continuous" \
	--max_temp=200 \
	--mask_lambda=1e-7 \
	--lr=1e-4 \
