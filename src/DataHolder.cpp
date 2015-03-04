#include "global.h"
#include "DataHolder.h"

template<class T> DataHolder<T>::DataHolder(int nx, int ny/*=1*/, int nz/*=1*/) : 
    nElements(nx, ny, nz)
    , totalElements(nx*ny*nz)
    , rawPtrCPU(0)
    , rawPtrGPU(0)
{
  rawPtrCPU = new T[totalElements];
  gpuErrorCheck(cudaMalloc(&rawPtrGPU, totalElements*sizeof(T))); 
  // Make sure both initialized arrays are zeroed.
  for (int i=0; i<totalElements; i++)
    rawPtrCPU[i] = 0;
  copyCPUtoGPU();
}

template<class T> DataHolder<T>::~DataHolder()
{
  delete rawPtrCPU;
  gpuErrorCheck(cudaFree(rawPtrGPU));
}

template<class T> void DataHolder<T>::copyCPUtoGPU()
{
  gpuErrorCheck(cudaMemcpy(rawPtrGPU
      , rawPtrCPU
      , totalElements*sizeof(T)
      , cudaMemcpyHostToDevice));
}

template<class T> void DataHolder<T>::copyGPUtoCPU()
{
  gpuErrorCheck(cudaMemcpy(rawPtrCPU
      , rawPtrGPU
      , totalElements*sizeof(T)
      , cudaMemcpyDeviceToHost));
}

// Force instantiation of DataHolder<> for datatype selected in datatype.h
template class DataHolder<datatype>;
