#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "FloatHolder.h"
#include "KernelLauncher.h"

int main()
{
  int N = 1<<20;
  float a = 2.0;
  FloatHolder fhx(N);
  FloatHolder fhy(N);
  KernelLauncher& kernelLauncher = KernelLauncher::instance();

  for (int i = 0; i < N; i++) 
  {
    fhx[i] = 1.0f;
    fhy[i] = 2.0f;
  }

  fhx.copyCPUtoGPU();
  fhy.copyCPUtoGPU();

  kernelLauncher.saxpy(fhx, fhy, a);

  fhy.copyGPUtoCPU();
 
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
  {
    maxError = std::max(maxError, std::abs(fhy[i]-4.0f));
  }
  printf("Max error: %f\n", maxError);
 
  return 0;
}  
  
