void cpu_mat_mul(float* M, float* N, float* P, int m, int p, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0;
      for (int k = 0; k < p; k++) {
        sum += M[i * p + k] * N[k * n + j];
      }
      P[i * n + j] = sum;
    }
  }
}
