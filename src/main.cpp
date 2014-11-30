#include <cstdlib>
#include <stdio.h>
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include "FloatHolder.h"
#include "KernelLauncher.h"

int main()
{
  int N = 1<<20;
  float a = 2.0;
  FloatHolder x(N);
  FloatHolder y(N)
  KernelLauncher& kernelLauncher = KernelLauncher::kl;

  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  x.copyCPUtoGPU();
  y.copyCPUtoGPU();

  kernelLauncher.saxpy(x, y, a);
 
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::max(maxError, std::abs(y[i]-4.0f));
  printf("Max error: %fn", maxError);
 
  return 0;
}  
  
