# 110M
# srun --exclusive -p EM --gres=gpu:4 ./runcc /shared/erc/getpTA/main/modelbin/stories110M.bin -m test -f ../in/gen_in_128.txt -o ./output_test_110M.txt

# 4B
srun --exclusive -p EM --gres=gpu:4 ./runcc /shared/erc/getpTA/main/modelbin/llama2-4b.bin -m test -f ../in/gen_in_128.txt -o ./output_test_4B.txt

# 7B
# srun --exclusive -p EM --gres=gpu:4 ./runcc /shared/erc/getpTA/main/modelbin/llama2-7b.bin -m test -f ../in/gen_in_128.txt -o ./output_test_7B.txt

# 13B
# srun --exclusive -p EM --gres=gpu:4 ./run13B /shared/erc/getpTA/main/modelbin/llama2-13b.bin -m test -f ../in/gen_in_128.txt -o ./output_test_13B.txt