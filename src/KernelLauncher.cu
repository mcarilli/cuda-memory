#include "KernelLauncher.h"
#include "FloatHolder.h"
// #include "cuPrintf.cu"
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

KernelLauncher KernelLauncher::kl;

__global__
void saxpyKernel(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  // printf("Thread number %d. i = %d\n", threadIdx.x, i);
  if (i < n) y[i] = a*x[i] + y[i];
}

void KernelLauncher::saxpy(FloatHolder& fhx, FloatHolder& fhy, float a)
{
  saxpyKernel<<<(fhx.nElements+255)/256, 256>>>(fhx.nElements
      , a
      , fhx.rawPtrGPU
      , fhy.rawPtrGPU);
  fhy.copyGPUtoCPU();
}

