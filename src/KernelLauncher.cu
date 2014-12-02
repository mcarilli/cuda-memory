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
void transposeNaiveKernel(int nx
    , int ny
    , float* in
    , float* out)
{
  int globalxindx = blockIdx.x*blockDim.x + threadIdx.x;
  int globalyindx = blockIdx.y*blockDim.y + threadIdx.y;
  /* printf("blockIdx.x %4d threadIdx.x %4d globalxindx %6d\nblockIdx.y %4d threadIdx.y %4d globalyindx %6d\n"
      , blockIdx.x
      , threadIdx.x
      , globalxindx
      , blockIdx.y
      , threadIdx.y
      , globalyindx); */
  if (globalxindx < nx && globalyindx < ny)
    out[nx*globalyindx+globalxindx] = in[ny*globalxindx+globalyindx];
    // out[ny*globalxindx+globalyindx] = in[nx*globalyindx+globalxindx];
}

// Inspired by http://www.cs.nyu.edu/manycores/cuda_many_cores.pdf
__global__ 
void transposeFastKernel(int nx
    , int ny
    , float* in
    , float* out)
{
  __shared__ float in_staging[BLOCKDIMY][BLOCKDIMX];
  __shared__ float out_staging[BLOCKDIMX][BLOCKDIMY];
  // For matrix | A B | with submatrices A, B, C, D, 
  //            | C D |
  // | A B |   = | A_T C_T |
  // | C D |_T   | B_T D_T |  
  // Reads to in_staging are coalesced.
  // Data is transposed within in_staging then written back
  // out to appropriate 32x32 block of out.
  int globalxoffset = blockIdx.x*blockDim.x;
  int globalyoffset = blockIdx.y*blockDim.y;
  int globalxindx = globalxoffset + threadIdx.x;
  int globalyindx = globalyoffset + threadIdx.y; 
  if (globalxindx < nx && globalyindx < ny)
    in_staging[threadIdx.y][threadIdx.x] = in[nx*globalyindx+globalxindx];
  __syncthreads();
  if (globalxindx < nx && globalyindx < ny)
  {
    out_staging[threadIdx.y][threadIdx.x] = in_staging[threadIdx.x][threadIdx.y];
  }
  __syncthreads();
  if (globalxindx < nx && globalyindx < ny)
    out[ny*(globalxoffset+threadIdx.y)+globalyoffset+threadIdx.x] = 
        out_staging[threadIdx.y][threadIdx.x];
}

void KernelLauncher::copy(FloatHolder& fhin, FloatHolder& fhout)
{
  cudaEventRecord(start);
  copyKernel
      // <<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      <<<dim3(1,(fhin.totalElements+BLOCKDIM-1)/BLOCKDIM),dim3(BLOCKDIM,1)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in copyKernel: %f ms\n", ms);
  printf("Effective throughput of copyKernel: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::saxpy(FloatHolder& fhx
    , FloatHolder& fhy
    , float a)
{
  cudaEventRecord(start);
  saxpyKernel
      <<<dim3(1,(fhx.totalElements+BLOCKDIM-1)/BLOCKDIM),dim3(BLOCKDIM,1)>>>
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
  printf("Elapsed time in transposeNaiveKernel: %f ms\n", ms);
  printf("Effective throughput of transposeNaiveKernel: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::transposeFast(FloatHolder& fhin, FloatHolder& fhout)
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
  printf("Elapsed time in tranposeFastKernel: %f\n", ms);
  printf("Effective throughput of transposeFastKernel: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

