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
  void copy(FloatHolder& fhin, FloatHolder& fhout);
  void saxpy(FloatHolder& fhx, FloatHolder& fhy, float a);
  void transposeNaive(FloatHolder& fhin, FloatHolder& fhout);
  void transposeFast(FloatHolder& fhin, FloatHolder& fhout);
  void transpose32PerThread(FloatHolder& fhin, FloatHolder& fhout);
  void matxvec(FloatHolder& fha, FloatHolder& fhv, FloatHolder& fhout);
  void matxmatNaive(FloatHolder& fha, FloatHolder& fhb, FloatHolder& fhout);
  void matxmatFast(FloatHolder& fha, FloatHolder& fhb, FloatHolder& fhbtranspose, FloatHolder& fhout);
};
