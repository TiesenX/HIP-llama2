# srun --exclusive -p EM --gres=gpu:4 ./runcc /shared/erc/getpTA/main/modelbin/llama2-4b.bin -m test -f ../in/gen_in_128.txt -o ./output_test.txt
srun --exclusive -p EM --gres=gpu:4 ./runcc /shared/erc/getpTA/main/modelbin/stories110M.bin -m test -f ../in/gen_in_128.txt -o ./output_test.txt