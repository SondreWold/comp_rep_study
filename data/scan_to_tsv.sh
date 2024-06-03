# Example use: scan_to_tsv.sh SCAN/simple_split/tasks_train_simple.txt train.tsv

input_file=$1
output_file=$2

sed -e "s/IN: //g" $input_file >> .tmp_scan.txt
sed -e "s/ OUT: /\t/g" .tmp_scan.txt > $output_file
