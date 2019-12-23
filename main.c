/* 
  Foundations of Parallel and Distributed Computing, Falls 2019.
  Instructor: Prof. Chao Yang @ Peking University.
  This code shows how to compute matrix multiplicaiton faster. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mat_mul.h"

int main(int argc, char **argv){
  int m = 500;  // default value
  int p = 1000;  // default value
  int n = 1500;  // default value
  if (argc > 1) {
      m = atoi(argv[1]); // user-specified value
      p = atoi(argv[2]); // user-specified value
      n = atoi(argv[3]); // user-specified value
  }
  printf("\nMatrix m: %d, p: %d, n: %d\n", m, p, n);

  srand(time(NULL));
  float* M = rand_mat(m, p);
  float* N = rand_mat(p, n);

  float* cpu_P = raw_mat(m, n);
  float* gpu_P = raw_mat(m, n);

  long long cpu_start_time = start_timer();
  cpu_mat_mul(M, N, cpu_P, m, p, n);
  long long cpu_time = stop_timer(cpu_start_time, "CPU");

  long long gpu_start_time = start_timer();
  gpu_mat_mul(M, N, gpu_P, m, p, n);
  long long gpu_time = stop_timer(gpu_start_time, "GPU");


  // Check the correctness of the GPU results
  int num_wrong = 0;
  for (int i = 0; i < m * n; i++) {
    if (fabs(cpu_P[i] - gpu_P[i]) > 0.000001) num_wrong++;
  }
	
  // Report the correctness results
  if (num_wrong) printf("GPU %d / %d values incorrect\n", num_wrong, m * n);
  else           printf("GPU all values correct\n");

}
