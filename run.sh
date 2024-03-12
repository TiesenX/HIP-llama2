srun --exclusive -p EM --gres=gpu:4 ./runcc ../modelbin/stories110M.bin -m test -f ../in/gen_in_128.txt -o ./output_test.txt
