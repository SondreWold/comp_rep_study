train_data_path="../../data/base_tasks/pcfgs_train.csv"
val_data_path="../../data/base_tasks/pcfgs_test.csv"
save_path="../model_checkpoints/pcfgs_base"
predictions_path="../predictions/pcfgs_base/"

python ../comp_rep/train.py \
	--train_data_path=$train_data_path \
	--val_data_path=$val_data_path \
	--lr 7e-5 \
	--hidden_size 512 \
	--layers 6 \
	--epochs 20 \
	--eval \
	--val_batch_size 64 \
	--eval \
	--save_path=$save_path \
	--predictions_path=$predictions_path \
