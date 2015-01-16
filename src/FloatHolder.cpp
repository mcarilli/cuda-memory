#include "FloatHolder.h"

FloatHolder::FloatHolder(int nx, int ny/*=1*/, int nz/*=1*/) : 
    nElements(nx, ny, nz)
    , totalElements(nx*ny*nz)
    , rawPtrCPU(0)
    , rawPtrGPU(0)
{
  rawPtrCPU = new float[totalElements];
  gpuErrorCheck(cudaMalloc(&rawPtrGPU, totalElements*sizeof(float))); 
}

FloatHolder::~FloatHolder()
{
  delete rawPtrCPU;
  gpuErrorCheck(cudaFree(rawPtrGPU));
}

void FloatHolder::copyCPUtoGPU()
{
  gpuErrorCheck(cudaMemcpy(rawPtrGPU
      , rawPtrCPU
      , totalElements*sizeof(float)
      , cudaMemcpyHostToDevice));
}

void FloatHolder::copyGPUtoCPU()
{
  gpuErrorCheck(cudaMemcpy(rawPtrCPU
      , rawPtrGPU
      , totalElements*sizeof(float)
      , cudaMemcpyDeviceToHost));
}    
