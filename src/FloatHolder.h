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
  // From http://stackoverflow.com/questions/14038589, helpful
  #define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  {
    if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
  }
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
  int nx() { return nElements.x; }
  int ny() { return nElements.y; }
  int nz() { return nElements.z; }
 
  friend class KernelLauncher;
};
