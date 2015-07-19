#pragma once
#include <cstdlib>
#include <cuda_runtime.h>

template<class T> class DataHolder;

template<class T> class KernelLauncher 
{
  static KernelLauncher<T> kl;
  cudaDeviceProp deviceProperties;
  unsigned int maxBlocks[3];
  KernelLauncher();
  ~KernelLauncher();
  KernelLauncher& operator=(KernelLauncher&); 
  KernelLauncher(const KernelLauncher&);
  cudaEvent_t start, stop;

  inline void startTiming();
  inline void finishTiming(const char* kernelName, DataHolder<T>& dh);

  public:
  static KernelLauncher<T>& instance() { return kl; }
  void copy(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void saxpy(DataHolder<T>& dhx, DataHolder<T>& dhy, T a);
  void transposeNaive(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void transposeFast(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void transposeFastNoBankConf(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void transpose4PerThread(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void matxmatNaive(DataHolder<T>& dha, DataHolder<T>& dhb, DataHolder<T>& dhout);
  void matxmatTiles(DataHolder<T>& dha, DataHolder<T>& dhb, DataHolder<T>& dhout);
  void reduceY(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void scan(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void unrollForLatency1(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void unrollForLatency2(DataHolder<T>& dhin, DataHolder<T>& dhout);
  void histogram(DataHolder<T>& dhin
      , DataHolder<T> blockhist
      , DataHolder<T> blockhistTranspose
      , DataHolder<T> hist );
  void bankconflicts();
};

