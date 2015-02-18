#include <cstdlib>
#include <iostream>
#include "KernelLauncher.h"
#include "FloatHolder.h"
// #include "cuPrintf.cu"

//For 1D kernels
#define BLOCKDIM 1024
// For 2D kernels, use 32x32 thread blocks = BLOCKDIM blocks/thread   
// This is the maximum allowable number on my machine, and         
// suffices for coalesced gmem reads.                             
#define BLOCKDIMX 32
#define BLOCKDIMY 32
 
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
//       << fhin.totalElements*sizeof(float)/ms/1e6 << " GB/s\n"; \
// } while(0) 

KernelLauncher KernelLauncher::kl;

KernelLauncher::KernelLauncher()
{
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
}

KernelLauncher::~KernelLauncher()
{
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void KernelLauncher::startTiming()
{
  cudaEventRecord(start);
}

void KernelLauncher::finishTiming(const char* kernelName, int totalElements)
{
  cudaEventRecord(stop); 
  cudaEventSynchronize(stop); 
  float ms = 0; 
  cudaEventElapsedTime(&ms, start, stop); 
  std::cout << "Average elapsed time in " << kernelName << ":  " << ms << " ms\n"; 
  std::cout << "Average effective data rate of " << kernelName << ":  " 
      << totalElements*sizeof(float)/ms/1e6 << " GB/s\n"; 
}

//Kernels

__global__ 
void copyKernel(int nx, int ny, float* in, float *out)
{
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int i = nx*globalyindx+globalxindx; 
  if (globalxindx < nx && globalyindx < ny)
    out[i] = in[i];
}

__global__
void saxpyKernel(int nx, int ny, float *x, float *y, float a)
{
  // Same structure as other kernels for consistent speed comparison. 
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int i = nx*globalyindx+globalxindx;
  if (globalxindx < nx && globalyindx < ny) 
    y[i] = a*x[i] + y[i];
  // printf("Thread number %d. y = %f, x = %f\n", threadIdx.x, y[i], x[i]);
}

__global__
void transposeNaiveKernel(int nx_out /*=ny_in*/
    , int ny_out /*=nx_in*/
    , float* in
    , float* out)
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
__global__ 
void transposeFastKernel(int nx_in
    , int ny_in
    , float* in
    , float* out)
{
  __shared__ float staging_in[BLOCKDIMY][BLOCKDIMX];
  __shared__ float staging_out[BLOCKDIMX][BLOCKDIMY];
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

__global__
void transposeFastNoBankConfKernel(int nx_in
    , int ny_in
    , float* in
    , float* out)
{
  __shared__ float staging_in[BLOCKDIMY][BLOCKDIMX+1];
  __shared__ float staging_out[BLOCKDIMX][BLOCKDIMY+1];
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

__global__
void matxmatNaiveKernel(int nx
      , int ny
      , float* a
      , float* b
      , float* out)
{
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int i = nx*globalyindx+globalxindx;
  float sum = 0;
  // #pragma unroll // Unrolling NOT useful here because nx not necessarily known at compile time
  for (int x=0; x<nx; x++)
    if (globalxindx < nx && globalyindx < ny)
      sum += a[nx*globalyindx+x]*b[ny*x+globalxindx];
  if (globalxindx < nx && globalyindx < ny)
    out[i] = sum;
}

__global__
void matxmatTilesKernel(int nx
      , int ny
      , float* a
      , float* b
      , float* out)
{
  // Square tiles in smem
  __shared__ float tileA[BLOCKDIMX][BLOCKDIMX+1];
  __shared__ float tileB[BLOCKDIMX][BLOCKDIMX+1];

  float sumOut = 0; // Holds output for this thread
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

    #pragma unroll // Unrolling could be useful here because BLOCKDIMX known at compile time
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

__global__ 
void reduceYBy2Kernel(int nx
    , int ny
    , int yStride
    , float* inout)
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

  float sum = 0;

  if (globalxindx < nx && globalyindx < ny)
  {
    sum += inout[nx*globalyindx+globalxindx];
    if ((globalyindx+yStride) < ny)
      sum += inout[nx*(globalyindx+yStride)+globalxindx];
    inout[nx*globalyindx+globalxindx] = sum;
  }
}

// Wrapper functions exposed by KernelLauncher

void KernelLauncher::copy(FloatHolder& fhin, FloatHolder& fhout)
{
  startTiming();
  copyKernel<<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY) \
      , dim3(BLOCKDIMX,BLOCKDIMY)>>>(fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  finishTiming("copyKernel", fhin.totalElements);
}

void KernelLauncher::saxpy(FloatHolder& fhx
    , FloatHolder& fhy
    , float a)
{
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    saxpyKernel
	<<<dim3((fhx.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhx.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(fhx.nx()
	, fhx.ny()
	, fhx.rawPtrGPU
	, fhy.rawPtrGPU
	, a);
  finishTiming("saxpy", fhx.totalElements);
}

void KernelLauncher::transposeNaive(FloatHolder& fhin, FloatHolder& fhout)
{
  startTiming(); 
  // for (int rep=0; rep<NREPS; rep++)
    transposeNaiveKernel
	<<<dim3((fhout.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhout.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(fhout.nx()
	, fhout.ny()
	, fhin.rawPtrGPU
	, fhout.rawPtrGPU);
  finishTiming("transposeNaive", fhin.totalElements);
}

void KernelLauncher::transposeFast(FloatHolder& fhin, FloatHolder& fhout)
{
  if (BLOCKDIMX != BLOCKDIMY) 
    printf("Warning:  transposeFast will fail if BLOCKDIMX (%d) != BLOCKDIMY (%d)"
        , BLOCKDIMX
        , BLOCKDIMY);
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    transposeFastKernel
	<<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(fhin.nx()
	, fhin.ny()
	, fhin.rawPtrGPU
	, fhout.rawPtrGPU);
  finishTiming("transposeFast", fhin.totalElements);
}

void KernelLauncher::transposeFastNoBankConf(FloatHolder& fhin, FloatHolder& fhout)
{
  if (BLOCKDIMX != BLOCKDIMY)
    printf("Warning:  transposeFastNoBankConf will fail if BLOCKDIMX (%d) != BLOCKDIMY (%d)"
        , BLOCKDIMX
        , BLOCKDIMY);
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    transposeFastNoBankConfKernel
	<<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(fhin.nx()
	, fhin.ny()
	, fhin.rawPtrGPU
	, fhout.rawPtrGPU);
  finishTiming("transposeFastNoBankConf", fhin.totalElements);
}

void KernelLauncher::transpose32PerThread(FloatHolder& fhin, FloatHolder& fhout)
{
  printf("Currently just runs transposeFast.  To be done later...");
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    transposeFastKernel
	<<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(fhin.nx()
	, fhin.ny()
	, fhin.rawPtrGPU
	, fhout.rawPtrGPU);
  finishTiming("transpose32PerThread", fhin.totalElements);
}

void KernelLauncher::matxmatNaive(FloatHolder& fha
    , FloatHolder& fhb
    , FloatHolder& fhout)
{
  startTiming();
  // for (int rep=0; rep<NREPS; rep++)
    matxmatNaiveKernel
	<<<dim3((fha.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fha.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	(fha.nx()
	, fha.ny()
	, fha.rawPtrGPU
	, fhb.rawPtrGPU
	, fhout.rawPtrGPU);
  finishTiming("matxmatNaive", fha.totalElements);
}

void KernelLauncher::matxmatTiles(FloatHolder& fha
    , FloatHolder& fhb
    , FloatHolder& fhout)
{
  startTiming();
  // Uses square tiles.  
  // printf("Using %dx%d grid of thread blocks\n",(fhb.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fha.ny()+BLOCKDIMX-1)/BLOCKDIMX);
  // for (int rep=0; rep<NREPS; rep++)
    matxmatTilesKernel
	<<<dim3((fhb.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fha.ny()+BLOCKDIMX-1)/BLOCKDIMX), dim3(BLOCKDIMX,BLOCKDIMX)>>>
	(fha.nx()
	, fha.ny()
	, fha.rawPtrGPU
	, fhb.rawPtrGPU
	, fhout.rawPtrGPU);
  finishTiming("matxmatTiles", fha.totalElements);
}

void KernelLauncher::reduceY(FloatHolder& fhin
    , FloatHolder& fhout)
{
  copyKernel
      <<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      // <<<dim3(1,(fhin.totalElements+BLOCKDIM-1)/BLOCKDIM),dim3(BLOCKDIM,1)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  startTiming();
  // for (int rep=0; rep<NREPS; rep++) 
  {
    int yStride = 1;
    int pass = 0;
    int nyRemaining = fhout.ny();
    while (nyRemaining >= 2)
    {
      // std::cout << "pass:  " << pass << std::endl;
      // std::cout << "yStride:  " << yStride << std::endl;
      // std::cout << "nyRemaining:  " << nyRemaining << std::endl;
      // std::cout << "Launching with: " << (nyRemaining+2*BLOCKDIMY-1)/(2*BLOCKDIMY) << " blocks in Y direction " << std::endl;
      reduceYBy2Kernel
	  <<<dim3((fhout.nx()+BLOCKDIMX-1)/BLOCKDIMX,(nyRemaining+2*BLOCKDIMY-1)/(2*BLOCKDIMY)), dim3(BLOCKDIMX,BLOCKDIMY)>>>
	  (fhout.nx()
	   , fhout.ny()
           , yStride
	   , fhout.rawPtrGPU);
      pass++;
      yStride *= 2; 
      nyRemaining /= 2;
    }
  }
  finishTiming("reduceYBy2Kernel", fhin.totalElements);
}
