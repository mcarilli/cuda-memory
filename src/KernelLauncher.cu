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
  __shared__ float staging_in[BLOCKDIMY+1][BLOCKDIMX];
  __shared__ float staging_out[BLOCKDIMX+1][BLOCKDIMY];
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
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  int i = nx*globalyindx+globalxindx;
  float sum = 0;
  for (int x=0; x<nx; x++)
    if (globalxindx < nx && globalyindx < ny)
      sum += a[nx*globalyindx+x]*b[ny*x+globalxindx];
  if (globalxindx < nx && globalyindx < ny)
    out[i] = sum;
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

void KernelLauncher::copy(FloatHolder& fhin, FloatHolder& fhout)
{
  cudaEventRecord(start);
  copyKernel
      <<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      // <<<dim3(1,(fhin.totalElements+BLOCKDIM-1)/BLOCKDIM),dim3(BLOCKDIM,1)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in copy: %f ms\n", ms);
  printf("Effective throughput of copy: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::saxpy(FloatHolder& fhx
    , FloatHolder& fhy
    , float a)
{
  cudaEventRecord(start);
  saxpyKernel
      <<<dim3((fhx.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhx.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      // <<<dim3(1,(fhx.totalElements+BLOCKDIM-1)/BLOCKDIM),dim3(BLOCKDIM,1)>>>
      (fhx.nx()
      , fhx.ny()
      , fhx.rawPtrGPU
      , fhy.rawPtrGPU
      , a);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in kernel saxpy: %f ms\n", ms);
  printf("Effective throughput of kernel saxpy: %f GB/s\n"
      , fhx.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::transposeNaive(FloatHolder& fhin, FloatHolder& fhout)
{ 
  cudaEventRecord(start);
  transposeNaiveKernel
      <<<dim3((fhout.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhout.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fhout.nx()
      , fhout.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in transposeNaive: %f ms\n", ms);
  printf("Effective throughput of transposeNaive: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::transposeFast(FloatHolder& fhin, FloatHolder& fhout)
{
  if (BLOCKDIMX != BLOCKDIMY) 
    printf("Warning:  transposeFast will fail if BLOCKDIMX (%d) != BLOCKDIMY (%d)"
        , BLOCKDIMX
        , BLOCKDIMY);
  cudaEventRecord(start);
  transposeFastKernel
      <<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in tranposeFast: %f ms\n", ms);
  printf("Effective throughput of transposeFast: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::transposeFastNoBankConf(FloatHolder& fhin, FloatHolder& fhout)
{
  if (BLOCKDIMX != BLOCKDIMY)
    printf("Warning:  transposeFastNoBankConf will fail if BLOCKDIMX (%d) != BLOCKDIMY (%d)"
        , BLOCKDIMX
        , BLOCKDIMY);
  cudaEventRecord(start);
  transposeFastNoBankConfKernel
      <<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in tranposeFastNoBankConf: %f ms\n", ms);
  printf("Effective throughput of transposeFastNoBankConf: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::transpose32PerThread(FloatHolder& fhin, FloatHolder& fhout)
{
  cudaEventRecord(start);
  transposeFastKernel
      <<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in tranpose32PerThread: %f ms\n", ms);
  printf("Effective throughput of transpose32PerThread: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::matxmatNaive(FloatHolder& fha
    , FloatHolder& fhb
    , FloatHolder& fhout)
{
  cudaEventRecord(start);
  matxmatNaiveKernel
      <<<dim3((fha.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fha.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fha.nx()
      , fha.ny()
      , fha.rawPtrGPU
      , fhb.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in matxmatNaive: %f ms\n", ms);
  printf("Effective throughput of matxmatNaive: %f GB/s\n"
      , fha.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::matxmatTiles(FloatHolder& fha
    , FloatHolder& fhb
    , FloatHolder& fhout)
{
  cudaEventRecord(start);
  matxmatTilesKernel
      <<<dim3((fha.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fha.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fha.nx()
      , fha.ny()
      , fha.rawPtrGPU
      , fhb.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in matxmatTiles: %f ms \n", ms);
  printf("Effective throughput of matxmatTiles: %f GB/s\n"
      , fha.totalElements*sizeof(float)/ms/1e6);
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
  cudaEventRecord(start);
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
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in reduceYBy2Kernel: %f ms\n", ms);
  printf("Effective throughput of reduceYBy2Kernel: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}
