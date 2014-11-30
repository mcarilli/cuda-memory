#pragma once
#include <cstdlib>
#include <stdio.h>

class KernelLauncher;

class FloatHolder
{
  int nElements;
  float *rawPtrCPU, *rawPtrGPU;

  public:
  FloatHolder(int n);
  ~FloatHolder();
  void copyCPUtoGPU();
  void copyGPUtoCPU();
  float* getrawPtrCPU(){ return rawPtrCPU; }
  float& operator[](int indx){ return rawPtrCPU[indx]; }
  
  friend class KernelLauncher;
};
