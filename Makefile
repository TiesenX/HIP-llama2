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

.PHONY: clean
clean:
	rm -f runcc
