#include "FloatHolder.h"
#include <cuda.h>
#include <cuda_runtime.h>

FloatHolder::FloatHolder(int nx, int ny/*=1*/, int nz/*=1*/) : 
    nElements(nx, ny, nz)
    , totalElements(nx*ny*nz)
    , rawPtrCPU(0)
    , rawPtrGPU(0)
{
  rawPtrCPU = new float[totalElements];
  cudaMalloc(&rawPtrGPU, totalElements*sizeof(float)); 
}

FloatHolder::~FloatHolder()
{
  delete rawPtrCPU;
  cudaFree(rawPtrGPU);
}

void FloatHolder::copyCPUtoGPU()
{
  cudaMemcpy(rawPtrGPU
      , rawPtrCPU
      , totalElements*sizeof(float)
      , cudaMemcpyHostToDevice);
}

void FloatHolder::copyGPUtoCPU()
{
  cudaMemcpy(rawPtrCPU
      , rawPtrGPU
      , totalElements*sizeof(float)
      , cudaMemcpyDeviceToHost);
}    
