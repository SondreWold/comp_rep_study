mkdir base_tasks
./scan_to_csv.sh SCAN/simple_split/tasks_train_simple.txt base_tasks/scan_train.csv
./scan_to_csv.sh SCAN/simple_split/tasks_test_simple.txt base_tasks/scan_test.csv

./pcfgs_to_csv.sh PCFGS/pcfgset/train.src PCFGS/pcfgset/train.tgt base_tasks/pcfgs_train.csv
./pcfgs_to_csv.sh PCFGS/pcfgset/dev.src PCFGS/pcfgset/dev.tgt base_tasks/pcfgs_dev.csv
./pcfgs_to_csv.sh PCFGS/pcfgset/test.src PCFGS/pcfgset/test.tgt base_tasks/pcfgs_test.csv



