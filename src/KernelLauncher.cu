#include "KernelLauncher.h"
#include "FloatHolder.h"
#include <cstdlib>
#include <iostream>
#include <cuda.h>

__global__
void saxpyKernel(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void KernelLauncher::saxpy(FloatHolder x, FloatHolder y, float a)
{
  saxpyKernel<<<(x.nElements+255)/256, 256>>>(x.nElements
      , a
      , x.rawPtrCPU
      , y.rawPtrCPU);
}

