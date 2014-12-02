#pragma once
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>

class KernelLauncher;

class FloatHolder
{
  dim3 nElements;
  int totalElements;
  float *rawPtrCPU, *rawPtrGPU;

  public:
  FloatHolder(int nx, int ny=1, int nz=1);
  ~FloatHolder();
  void copyCPUtoGPU();
  void copyGPUtoCPU();
  float* getrawPtrCPU(){ return rawPtrCPU; }
  float& operator[](int indx){ return rawPtrCPU[indx]; }
  float& operator()(int z, int y, int x)
  { 
    return rawPtrCPU[nElements.x*nElements.y*z + nElements.x*y + x]; 
  }
  float nx() { return nElements.x; }
  float ny() { return nElements.y; }
  float nz() { return nElements.z; }
 
  friend class KernelLauncher;
};
