CC = gcc
CFLAGS = -std=c99

NVCC = nvcc
NVCC_FLAGS = -std=c++11 -O3 -Wno-deprecated-gpu-targets

LIBRARIES = -L/usr/local/cuda/lib64 -lcudart -lm

mat_mul: main.o cpu_mat_mul.o gpu_mat_mul.o 
	$(CC) $^ -o $@ $(LIBRARIES)

main.o: main.c
	$(CC) $(CFLAGS) -c $^ -o $@

cpu_mat_mul.o: mat_mul.c
	$(CC) $(CFLAGS) -c $^ -o $@

gpu_mat_mul.o: mat_mul.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

clean:
	rm -f *.o mat_mul
	
