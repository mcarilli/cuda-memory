#include <cstdlib>
#include <iostream>
#include <vector>
#include "KernelLauncher.h"
#include "global.h"
#include "DataHolder.h"
// #include "cuPrintf.cu"

#define NREPS 10

// More trouble than it's worth.
// #define TIMINGINFO(kernelName, griddim, blockdim, ...) do { \
//   cudaEventRecord(start); \
//   kernelName##<<<griddim,blockdim>>>(__VA_ARGS__); \
//   cudaEventRecord(stop); \
//   cudaEventSynchronize(stop); \
//   float ms = 0; \
//   cudaEventElapsedTime(&ms, start, stop); \
//   std::cout << "Average elapsed time in " << #kernelName << ":  " << ms << " ms\n"; \
//   std::cout << "Average effective data rate of " << #kernelName ":  " \
//       << dhin.totalElements*sizeof(float)/ms/1e6 << " GB/s\n"; \
// } while(0) 

// Initialize static data members
template<class T> KernelLauncher<T> KernelLauncher<T>::kl;

template<class T> KernelLauncher<T>::KernelLauncher()
{
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&deviceProperties,device);
  maxBlocks[0] = deviceProperties.maxGridSize[0];
  maxBlocks[1] = deviceProperties.maxGridSize[1];
  maxBlocks[2] = deviceProperties.maxGridSize[2];
  std::cout << "device:  " << device << std::endl;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}

template<class T> KernelLauncher<T>::~KernelLauncher()
{
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

template<class T> void KernelLauncher<T>::startTiming()
{
  cudaEventRecord(start);
}

template<class T> void KernelLauncher<T>::finishTiming(const char* kernelName, DataHolder<T>& dh)
{
  cudaEventRecord(stop); 

  if ( cudaEventSynchronize(stop) != cudaSuccess )
    std::cout << "After " << kernelName << ":  " << cudaGetErrorString(cudaGetLastError()) << "\n"; 

  float ms = 0; 
  cudaEventElapsedTime(&ms, start, stop); 
  std::cout << "sizeof(datatype):  " << sizeof(T) << "\n"; 
  std::cout << "Average elapsed time in " << kernelName << ":  " << ms << " ms\n"; 
  std::cout << "Average effective data rate of " << kernelName << ":  " 
      << dh.totalElements*sizeof(T)/ms/1e6 << " GB/s\n"; 
}

//Kernels

template<class T> __global__ void 
copyKernel(int nx, int ny, T* in, T *out)
{
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int i = nx*globalyindx+globalxindx; 
  if (globalxindx < nx && globalyindx < ny)
    out[i] = in[i];
}

template<class T> __global__ void 
saxpyKernel(int nx, int ny, T *x, T *y, T a)
{
  // Same structure as other kernels for consistent speed comparison. 
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int i = nx*globalyindx+globalxindx;
  if (globalxindx < nx && globalyindx < ny) 
    y[i] = a*x[i] + y[i];
  // printf("Thread number %d. y = %f, x = %f\n", threadIdx.x, y[i], x[i]);
}

template<class T> __global__ void 
transposeNaiveKernel(int nx_out /*=ny_in*/
    , int ny_out /*=nx_in*/
    , T* in
    , T* out)
{
  int globalxindx_out = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx_out = blockIdx.y*blockDim.y + threadIdx.y;
  /* printf("blockIdx.x %4d threadIdx.x %4d globalxindx %6d\nblockIdx.y %4d threadIdx.y %4d globalyindx %6d\n"
      , blockIdx.x
      , threadIdx.x
      , globalxindx_out
      , blockIdx.y
      , threadIdx.y
      , globalyindx_out); */
  if (globalxindx_out < nx_out && globalyindx_out < ny_out)
    out[nx_out*globalyindx_out+globalxindx_out] = in[ny_out*globalxindx_out+globalyindx_out];
    // out[ny*globalxindx+globalyindx] = in[nx*globalyindx+globalxindx];
}

// Inspired by http://www.cs.nyu.edu/manycores/cuda_many_cores.pdf
template<class T> __global__ void 
transposeFastKernel(int nx_in
    , int ny_in
    , T* in
    , T* out)
{
  __shared__ T staging_in[BLOCKDIMY][BLOCKDIMX];
  __shared__ T staging_out[BLOCKDIMX][BLOCKDIMY];
  // For matrix | A B | with submatrices A, B, C, D, 
  //            | C D |
  // | A B |   = | A_T C_T |
  // | C D |_T   | B_T D_T |  
  // Reads to in_staging are coalesced.
  // Data is transposed within in_staging then written back
  // out to appropriate 32x32 block of out.
  int globalxoffset_in = blockIdx.x*blockDim.x;
  int globalyoffset_in = blockIdx.y*blockDim.y;
  int globalxindx_in = globalxoffset_in + threadIdx.x;
  int globalyindx_in = globalyoffset_in + threadIdx.y; 
  if (globalxindx_in < nx_in && globalyindx_in < ny_in)
    staging_in[threadIdx.y][threadIdx.x] = in[nx_in*globalyindx_in+globalxindx_in];
  __syncthreads();
  if (globalxindx_in < nx_in && globalyindx_in < ny_in)
  {
    staging_out[threadIdx.x][threadIdx.y] = staging_in[threadIdx.y][threadIdx.x];
  }
  __syncthreads();
  if (globalxindx_in < nx_in && globalyindx_in < ny_in)
    out[ny_in*(globalxoffset_in+threadIdx.y)+globalyoffset_in+threadIdx.x] = 
        staging_out[threadIdx.y][threadIdx.x];
}

template<class T> __global__ void 
transposeFastNoBankConfKernel(int nx_in
    , int ny_in
    , T* in
    , T* out)
{
  __shared__ T staging_in[BLOCKDIMY][BLOCKDIMX+1];
  __shared__ T staging_out[BLOCKDIMX][BLOCKDIMY+1];
  // For matrix | A B | with submatrices A, B, C, D, 
  //            | C D |
  // | A B |   = | A_T C_T |
  // | C D |_T   | B_T D_T |  
  // Reads to in_staging are coalesced.
  // Data is transposed within in_staging then written back
  // out to appropriate 32x32 block of out.
  int globalxoffset_in = blockIdx.x*blockDim.x;
  int globalyoffset_in = blockIdx.y*blockDim.y;
  int globalxindx_in = globalxoffset_in + threadIdx.x;
  int globalyindx_in = globalyoffset_in + threadIdx.y;
  if (globalxindx_in < nx_in && globalyindx_in < ny_in)
    staging_in[threadIdx.y][threadIdx.x] = in[nx_in*globalyindx_in+globalxindx_in];
  __syncthreads();
  if (globalxindx_in < nx_in && globalyindx_in < ny_in)
  {
    staging_out[threadIdx.x][threadIdx.y] = staging_in[threadIdx.y][threadIdx.x];
  }
  __syncthreads();
  if (globalxindx_in < nx_in && globalyindx_in < ny_in)
    out[ny_in*(globalxoffset_in+threadIdx.y)+globalyoffset_in+threadIdx.x] =
        staging_out[threadIdx.y][threadIdx.x];
}

template<class T> __global__ void 
matxmatNaiveKernel(int nx
      , int ny
      , T* a
      , T* b
      , T* out)
{
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int i = nx*globalyindx+globalxindx;
  T sum = 0;
  // #pragma unroll // Unrolling NOT useful here because nx not necessarily known at compile time
  for (int x=0; x<nx; x++)
    if (globalxindx < nx && globalyindx < ny)
      sum += a[nx*globalyindx+x]*b[ny*x+globalxindx];
  if (globalxindx < nx && globalyindx < ny)
    out[i] = sum;
}

template<class T> __global__ void 
matxmatTilesKernel(int nx
      , int ny
      , T* a
      , T* b
      , T* out)
{
  // Square tiles in smem
  __shared__ T tileA[BLOCKDIMX][BLOCKDIMX+1];
  __shared__ T tileB[BLOCKDIMX][BLOCKDIMX+1];

  T sumOut = 0; // Holds output for this thread
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int tileATopLeftxindx = 0;
  int tileATopLeftyindx = 0;
  int tileBTopLeftxindx = 0;
  int tileBTopLeftyindx = 0;
  bool inRange = (globalxindx < nx && globalyindx < ny);

  for (int tileindx=0; tileindx<(nx+BLOCKDIMX-1)/BLOCKDIMX; tileindx++)
  {
    tileATopLeftxindx = tileindx*BLOCKDIMX;
    tileATopLeftyindx = blockIdx.y*blockDim.y;
    tileBTopLeftxindx = blockIdx.x*blockDim.x;
    tileBTopLeftyindx = tileATopLeftxindx;

    // Load square tiles into smem
    if(inRange)
    {
      // Loads are coalesced for both tileA and tileB.
      tileA[threadIdx.y][threadIdx.x] = a[ny*(tileATopLeftyindx+threadIdx.y)+tileATopLeftxindx+threadIdx.x]; 
      tileB[threadIdx.y][threadIdx.x] = b[nx*(tileBTopLeftyindx+threadIdx.y)+tileBTopLeftxindx+threadIdx.x];
    }

    __syncthreads();

    // #pragma unroll // Unrolling potentially useful here because BLOCKDIMX known at compile time
    for (int x=0; x<BLOCKDIMX; x++)
      if(inRange)
	sumOut += tileA[threadIdx.y][x]*tileB[x][threadIdx.x];      

    __syncthreads(); 

   // if (nx*globalyindx+globalxindx == 0/*threadIdx.x == 0 && threadIdx.y == 0*/)
   //   printf("blockIdx.x: %d, blockIdx.y: %d, tileindx: %d, tileA[0][1]: %f, tileB[0][1]: %f, \ntileOut[1][0]: %f, globalxindx: %d, globalyindx: %d, # of tileindxs: %d\n\n"
   //       , blockIdx.x
   //       , blockIdx.y
   //       , tileindx
   //       , tileA[0][1]
   //       , tileB[0][1]
   //       , tileOut[1][0]
   //       , globalxindx
   //       , globalyindx
   //       , (nx+BLOCKDIMX-1)/BLOCKDIMX);

  }
  if(inRange)
    out[nx*globalyindx+globalxindx] = sumOut; 
}

template<class T> __global__ void 
reduceYBy2Kernel(int nx
    , int ny
    , int yStride
    , T* inout)
{
  // Reduce like this
  //  a b  ->  a+c b+d  // a+c was produced by thread 1, b+d by thread 2, etc 
  //  c d      garbage  
  //  e f      e+g f+h
  //  g h      garbage
  // Memory accesses are x-contiguous => coalesced 
  // for good choice of blockDim.x
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = 2*yStride*(blockIdx.y*blockDim.y + threadIdx.y);

  T sum = 0;

  if (globalxindx < nx && globalyindx < ny)
  {
    sum += inout[nx*globalyindx+globalxindx];
    if ((globalyindx+yStride) < ny)
      sum += inout[nx*(globalyindx+yStride)+globalxindx];
    inout[nx*globalyindx+globalxindx] = sum;
  }
}


// Based on https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
#define PADDED(n) (n+(n>>5)) // 32 smem banks; n>>5 = n/32 
                             // Add one bank of padding between every 32 stored elements,
                             // as for the 2D smem access patterns above.
template<class T> __global__ void
scanWithinBlocksKernel(int nx
    , int ngrids
    , T* in
    , T* out
    , T* sum)
{
  extern __shared__ T scratch[];
  // Each block scans over 2*blockDim.x elements
  // Gmem arrays are padded to SCANSECTION = 2*blockDim.x
  for (int grid = 0; grid < ngrids; grid++)
  {
    int globalxindx = 2*blockIdx.x*blockDim.x + threadIdx.x + grid*blockDim.x*gridDim.x;

    // Load data from gmem.  Reads are coalesced.
    scratch[PADDED(threadIdx.x)] = in[globalxindx];
    scratch[PADDED(threadIdx.x+blockDim.x)] = in[globalxindx+blockDim.x];
    
    // Perform binary tree-style reduction into rightmost element of array
    int offset = 1;
    for (int activeThreads = blockDim.x; activeThreads > 0; activeThreads >>= 1)
    {
      __syncthreads();

      if (threadIdx.x < activeThreads)
      {
	int src = offset*(2*threadIdx.x+1)-1; // dst-offset
	int dst = offset*(2*threadIdx.x+2)-1; // 2*offset*(threadIdx.x+1)-1 
	scratch[PADDED(dst)] += scratch[PADDED(src)]; 
      }
      offset <<= 1;
    }

    // Make sure all threads have same view of memory
    // (have flushed writes to smem)
    __syncthreads();

    // Store total sum of this scan section
    // (to be added to later sections) 
    if (threadIdx.x == 0)
    {
      sum[blockIdx.x+grid*gridDim.x] = scratch[PADDED(SCANSECTION-1)];
    }   
 
    // Zero the last element
    if (threadIdx.x == 0)
      scratch[PADDED(SCANSECTION-1)] = 0;
   
    // Perform tree-style down-sweep, moving sums one index to the right
    for (int activeThreads = 1; activeThreads < SCANSECTION; activeThreads <<= 1)
    {
      offset >>= 1;
      __syncthreads();
      if (threadIdx.x < activeThreads)
      {
	int left = offset*(2*threadIdx.x+1)-1;
	int right = offset*(2*threadIdx.x+2)-1;
	T temp = scratch[PADDED(left)];
	scratch[PADDED(left)] = scratch[PADDED(right)];
	scratch[PADDED(right)] += temp; 
      } 
    }

    // Make sure all threads have flushed their writes to smem
    __syncthreads();

    // Output data to gmem
    out[globalxindx] = scratch[PADDED(threadIdx.x)];
    out[globalxindx+blockDim.x] = scratch[PADDED(threadIdx.x+blockDim.x)];
  }
}


template<class T> __global__ void
scanApplySum(int nx
    , int ngrids
    , T* __restrict__ out
    , const T* __restrict__ sum) // Enable use of constant cache for sum
{
  for (int grid = 0; grid < ngrids; grid++)
  {
    int globalxindx = 2*blockIdx.x*blockDim.x + threadIdx.x + grid*blockDim.x*gridDim.x;
    out[globalxindx] += sum[blockIdx.x];
    out[globalxindx+blockDim.x] += sum[blockIdx.x];
  }
}

// The sums array needs to be built up recursively from 1 block.
template<class T> void scanRecursive(unsigned int nx
    , unsigned int maxBlocks
    , int currentDepth
    , T* in
    , T* out
    , std::vector<T*> sums
    , std::vector<T*> sumsOut)
{
  std::cout << "nx at this level = PADTOSECDIM(nx/SCANSECTION) = " << nx << std::endl;

  // nx should be a multiple of SCANSECTION.
  unsigned int ngrids = (nx/SCANSECTION+maxBlocks-1)/maxBlocks;
  scanWithinBlocksKernel
     <<<(nx/SCANSECTION>maxBlocks)?maxBlocks:nx/SCANSECTION, SCANSECTION>>1, (PADDED(SCANSECTION)+1)*sizeof(T)>>>
      (nx
      , ngrids
      , in
      , out
      , sums[currentDepth]);

  if ( nx <= SCANSECTION)
    return;

  scanRecursive(PADTOSECDIM(nx/SCANSECTION)
      , maxBlocks
      , currentDepth+1
      , sums[currentDepth]
      , sumsOut[currentDepth]
      , sums
      , sumsOut);

  scanApplySum
    <<<(nx/SCANSECTION>maxBlocks)?maxBlocks:nx/SCANSECTION, SCANSECTION>>1, (PADDED(SCANSECTION)+1)*sizeof(T)>>>
    (nx
    , ngrids
    , out
    , sumsOut[currentDepth]);
}
// Wrapper functions exposed by KernelLauncher<T>

template<class T> void KernelLauncher<T>::copy(DataHolder<T>& dhin
    , DataHolder<T>& dhout)
{
  startTiming();
  copyKernel<<<dim3((dhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dhin.ny()+BLOCKDIMY-1)/BLOCKDIMY) \
      , dim3(BLOCKDIMX,BLOCKDIMY)>>>(dhin.nx()
      , dhin.ny()
      , dhin.rawPtrGPU
      , dhout.rawPtrGPU);
  finishTiming("copyKernel", dhin);
}

template<class T> void KernelLauncher<T>::saxpy(DataHolder<T>& dhx
    , DataHolder<T>& dhy
    , T a)
{
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    saxpyKernel
	<<<dim3((dhx.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dhx.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(dhx.nx()
	, dhx.ny()
	, dhx.rawPtrGPU
	, dhy.rawPtrGPU
	, a);
  finishTiming("saxpy", dhx);
}

template<class T> void KernelLauncher<T>::transposeNaive(DataHolder<T>& dhin
    , DataHolder<T>& dhout)
{
  startTiming(); 
  // for (int rep=0; rep<NREPS; rep++)
    transposeNaiveKernel
	<<<dim3((dhout.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dhout.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(dhout.nx()
	, dhout.ny()
	, dhin.rawPtrGPU
	, dhout.rawPtrGPU);
  finishTiming("transposeNaive", dhin);
}

template<class T> void KernelLauncher<T>::transposeFast(DataHolder<T>& dhin
    , DataHolder<T>& dhout)
{
  if (BLOCKDIMX != BLOCKDIMY) 
    printf("Warning:  transposeFast will fail if BLOCKDIMX (%d) != BLOCKDIMY (%d)"
        , BLOCKDIMX
        , BLOCKDIMY);
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    transposeFastKernel
	<<<dim3((dhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(dhin.nx()
	, dhin.ny()
	, dhin.rawPtrGPU
	, dhout.rawPtrGPU);
  finishTiming("transposeFast", dhin);
}

template<class T> void KernelLauncher<T>::transposeFastNoBankConf(DataHolder<T>& dhin
    , DataHolder<T>& dhout)
{
  if (BLOCKDIMX != BLOCKDIMY)
    printf("Warning:  transposeFastNoBankConf will fail if BLOCKDIMX (%d) != BLOCKDIMY (%d)"
        , BLOCKDIMX
        , BLOCKDIMY);
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    transposeFastNoBankConfKernel
	<<<dim3((dhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(dhin.nx()
	, dhin.ny()
	, dhin.rawPtrGPU
	, dhout.rawPtrGPU);
  finishTiming("transposeFastNoBankConf", dhin);
}

template<class T> void KernelLauncher<T>::transpose32PerThread(DataHolder<T>& dhin, DataHolder<T>& dhout)
{
  printf("Currently just runs transposeFast.  To be done later...");
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    transposeFastKernel
	<<<dim3((dhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(dhin.nx()
	, dhin.ny()
	, dhin.rawPtrGPU
	, dhout.rawPtrGPU);
  finishTiming("transpose32PerThread", dhin);
}

template<class T> void KernelLauncher<T>::matxmatNaive(DataHolder<T>& dha
    , DataHolder<T>& dhb
    , DataHolder<T>& dhout)
{
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    matxmatNaiveKernel
	<<<dim3((dha.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dha.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(dha.nx()
	, dha.ny()
	, dha.rawPtrGPU
	, dhb.rawPtrGPU
	, dhout.rawPtrGPU);
  finishTiming("matxmatNaive", dha);
}

template<class T> void KernelLauncher<T>::matxmatTiles(DataHolder<T>& dha
    , DataHolder<T>& dhb
    , DataHolder<T>& dhout)
{
  startTiming();
  // Uses square tiles.  
  // printf("Using %dx%d grid of thread blocks\n",(dhb.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dha.ny()+BLOCKDIMX-1)/BLOCKDIMX);
  // for (int rep=0; rep<NREPS; rep++)
    matxmatTilesKernel
	<<<dim3((dhb.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dha.ny()+BLOCKDIMX-1)/BLOCKDIMX), dim3(BLOCKDIMX,BLOCKDIMX)>>>
	(dha.nx()
	, dha.ny()
	, dha.rawPtrGPU
	, dhb.rawPtrGPU
	, dhout.rawPtrGPU);
  finishTiming("matxmatTiles", dha);
}

template<class T> void KernelLauncher<T>::reduceY(DataHolder<T>& dhin
    , DataHolder<T>& dhout)
{
  copyKernel
      <<<dim3((dhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(dhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      // <<<dim3(1,(dhin.totalElements+BLOCKDIM-1)/BLOCKDIM),dim3(BLOCKDIM,1)>>>
      (dhin.nx()
      , dhin.ny()
      , dhin.rawPtrGPU
      , dhout.rawPtrGPU);
  startTiming();
  // for (int rep=0; rep<NREPS; rep++) 
  {
    int yStride = 1;
    int pass = 0;
    int nyRemaining = dhout.ny();
    while (nyRemaining >= 2)
    {
      // std::cout << "pass:  " << pass << std::endl;
      // std::cout << "yStride:  " << yStride << std::endl;
      // std::cout << "nyRemaining:  " << nyRemaining << std::endl;
      // std::cout << "Launching with: " << (nyRemaining+2*BLOCKDIMY-1)/(2*BLOCKDIMY) << " blocks in Y direction " << std::endl;
      reduceYBy2Kernel
	  <<<dim3((dhout.nx()+BLOCKDIMX-1)/BLOCKDIMX,(nyRemaining+2*BLOCKDIMY-1)/(2*BLOCKDIMY)), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	  (dhout.nx()
	  , dhout.ny()
          , yStride
	  , dhout.rawPtrGPU);
      pass++;
      yStride *= 2; 
      nyRemaining /= 2;
    }
  }
  finishTiming("reduceYBy2Kernel", dhin);
}

template<class T> void KernelLauncher<T>::scan(DataHolder<T>& dhin
    , DataHolder<T>& dhout)
{
  // Allocate memory for sums.  I don't consider the allocation as part of the timing test. 
  std::vector<DataHolder<T>*> sumsVec;
  std::vector<DataHolder<T>*> sumsOutVec;

  int n=dhin.nx()/SCANSECTION;

  while (n>1)
  {
     // Sums arrays are padded to a multiple of block size.
     sumsVec.push_back(new DataHolder<T>(PADTOSECDIM(n)));
     sumsOutVec.push_back(new DataHolder<T>(PADTOSECDIM(n)));
     n = DIVUP(n,SCANSECTION);
  }

  // Always allocate at least one sums array.
  sumsVec.push_back(new DataHolder<T>(PADTOSECDIM(1)));
  sumsOutVec.push_back(new DataHolder<T>(PADTOSECDIM(1)));

  std::vector<T*> sums(sumsVec.size());
  std::vector<T*> sumsOut(sumsOutVec.size());

  for (int n=0; n<sumsVec.size(); n++)
  {
    sums[n] = sumsVec[n]->rawPtrGPU;
    sumsOut[n] = sumsOutVec[n]->rawPtrGPU;
  }

  startTiming();
  scanRecursive(dhin.nx() /* int nx */
    , maxBlocks[0] /* int maxBlocks */
    , 0 /* int currentDepth */
    , dhin.rawPtrGPU /* T* in */
    , dhout.rawPtrGPU /* T* out */
    , sums /* std::vector<T*> sums */
    , sumsOut /* std::vector<T*> sumsOut */); 
  finishTiming("scanKernel", dhin);

  for (int i=0; i<sumsVec.size(); i++)
  {
    delete sumsVec[i];
    delete sumsOutVec[i];
  }
}

// Force instantiation of KernelLauncher<> for datatype selected in datatype.h
template class KernelLauncher<datatype>;

