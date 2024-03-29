# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = hipcc

# the most basic way of building that is most likely to work on most systems
.PHONY: runcpu
runcpu: run.cc
	$(CC) -o runcc run.cc -O2 --offload-arch=gfx908 -fopenmp

.PHONY: rungpu
rungpu: run.cc
	$(CC) -DUSE_GPU -o runcc run.cc -O2 --offload-arch=gfx908 -fopenmp

.PHONY: run13B
run13B: run_13B.cc
	$(CC) -DUSE_GPU -o run13B run_13B.cc -O2 --offload-arch=gfx908 -fopenmp

.PHONY: test
test: testing.cc run.cc
	$(CC) -DKERNEL_TEST -o testcc testing.cc run.cc -O2 --offload-arch=gfx908 -fopenmp

.PHONY: clean
clean:
	rm -f runcc testcc run13B *.csv *.db output_test*.txt *.json result*.txt   