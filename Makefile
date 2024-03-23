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

.PHONY: runselect
runselect: run_select.cc
	$(CC) -DUSE_GPU -o runselect run_select.cc -O2 --offload-arch=gfx908 -fopenmp

.PHONY: runschedule_4B
runschedule_4B: run_schedule_4B.cc
	$(CC) -DUSE_GPU -o runschedule_4B run_schedule_4B.cc -O3 --offload-arch=gfx908 -fopenmp

.PHONY: runschedule_7B
runschedule_7B: run_schedule_7B.cc
	$(CC) -DUSE_GPU -o runschedule_7B run_schedule_7B.cc -O3 --offload-arch=gfx908 -fopenmp

.PHONY: run13B
run13B: run_13B.cc
	$(CC) -DUSE_GPU -o run13B run_13B.cc -O2 --offload-arch=gfx908 -fopenmp

.PHONY: test
test: testing.cc run.cc
	$(CC) -DKERNEL_TEST -o testcc testing.cc run.cc -O2 --offload-arch=gfx908 -fopenmp

.PHONY: clean
clean:
	rm -f runcc testcc run13B runschedule_7B runschedule_4B *.csv *.db output_test*.txt *.json result*.txt   