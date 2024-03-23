# 4B
# srun --exclusive -p EM --gres=gpu:4 ./runschedule_4B /shared/erc/getpTA/main/modelbin/llama2-4b.bin -m test -f ../in/gen_in_1024.txt -o ./output_test_4B.txt

# 7B
srun --exclusive -p EM --gres=gpu:4 ./runschedule_7B /shared/erc/getpTA/main/modelbin/llama2-7b.bin -m test -f ../in/gen_in_512.txt -o ./output_test_7B.txt

# 13B
# srun --exclusive -p EM --gres=gpu:4 ./run13B /shared/erc/getpTA/main/modelbin/llama2-13b.bin -m test -f ../in/gen_in_128.txt -o ./output_test_13B.txt

