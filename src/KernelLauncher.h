#pragma once
#include <cstdlib>
#include <cuda_runtime.h>

class FloatHolder;

class KernelLauncher 
{
  static KernelLauncher kl;
  KernelLauncher();
  ~KernelLauncher();
  KernelLauncher& operator=(KernelLauncher&); 
  KernelLauncher(const KernelLauncher&);
  cudaEvent_t start, stop;
  public:
  static KernelLauncher& instance() { return kl; }
  void saxpy(FloatHolder& fhx, FloatHolder& fhy, float a);
  void transposeNaive(FloatHolder& fhin, FloatHolder& fhout);
  void transposeFast(FloatHolder& fhin, FloatHolder& fhout);
};
