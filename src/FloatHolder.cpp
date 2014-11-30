#include "FloatHolder.h"
#include <cuda.h>
#include <cuda_runtime.h>

FloatHolder::FloatHolder(int n) : nElements(n)
    , rawPtrCPU(0)
    , rawPtrGPU(0)
{
  rawPtrCPU = new float[nElements];
  cudaMalloc(&rawPtrGPU, nElements*sizeof(float)); 
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
      , nElements*sizeof(float)
      , cudaMemcpyHostToDevice);
}

void FloatHolder::copyGPUtoCPU()
{
  cudaMemcpy(rawPtrCPU
      , rawPtrGPU
      , nElements*sizeof(float)
      , cudaMemcpyDeviceToHost);
}    
