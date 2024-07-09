mkdir function_tasks
for function in "copy" "reverse" "shift" "echo" "swap_first_last" "repeat" "append" "prepend" "remove_first" "remove_second"
do
	echo Generating data for function: $function
	python ../comp-rep/comp_rep/data_prep/create_function_data.py --nr_samples 10000 --output_path ${function} --task $function
	mkdir function_tasks/${function}
	sed 's/\t/;/g' ${function}.txt > t_${function}.txt
	split -l 8000 t_${function}.txt
	mv xaa function_tasks/${function}/train.csv
	mv xab function_tasks/${function}/test.csv
	rm ${function}.txt
	rm t_${function}.txt
done
