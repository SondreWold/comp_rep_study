# Example use: pcfgs_to_tsv.sh PCFGS/pcfgset/train.src pcfgs_to_tsv.sh PCFGS/pcfgset/train.tgt train.csv
src_file=$1
tgt_file=$2
save_path=$3

paste -d ";" $1 $2 > $save_path
