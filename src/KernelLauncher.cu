#include <cstdlib>
#include <iostream>
#include "KernelLauncher.h"
#include "FloatHolder.h"
// #include "cuPrintf.cu"

// For 2D kernels, use 32x32 thread blocks = 1024 blocks/thread   
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

// Based on http://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/
__global__
void saxpyKernel(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
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
  // special treatment for main diagonal not worth thread divergence.
  if (globalxindx < nx && globalyindx < ny)
    out[ny*globalxindx+globalyindx] = in[nx*globalyindx+globalxindx];
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

void KernelLauncher::saxpy(FloatHolder& fhx
    , FloatHolder& fhy
    , float a)
{
  cudaEventRecord(start);
  saxpyKernel<<<(fhx.totalElements+255)/256, 256>>>(fhx.totalElements
      , a
      , fhx.rawPtrGPU
      , fhy.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in kernel saxpy: %f\n", ms);
  printf("Effective throughput of kernel saxpy: %f GB/s\n"
      , fhx.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::transposeNaive(FloatHolder& fhin, FloatHolder& fhout)
{ 
  cudaEventRecord(start);
  transposeNaiveKernel
      <<<dim3((fhin.nx()+31)/32,(fhin.ny()+31)/32), dim3(32,32)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in kernel transposeNaive: %f\n", ms);
  printf("Effective throughput of kernel transposeNaive: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::transposeFast(FloatHolder& fhin, FloatHolder& fhout)
{
  cudaEventRecord(start);
  transposeFastKernel
      <<<dim3((fhin.nx()+31)/32,(fhin.ny()+31)/32), dim3(32,32)>>>
      (fhin.nx()
      , fhin.ny()
      , fhin.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in kernel tranposeFast: %f\n", ms);
  printf("Effective bandwidth of kernel transposeFast: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

