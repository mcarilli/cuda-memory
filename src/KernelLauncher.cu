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
    sum += a[nx*globalyindx+x]*b[ny*x+globalxindx];
  out[i] = sum;
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
  printf("Elapsed time in tranposeFast: %f\n", ms);
  printf("Effective throughput of transposeFast: %f GB/s\n"
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
  printf("Elapsed time in tranpose32PerThread: %f\n", ms);
  printf("Effective throughput of transpose32PerThread: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::matxvec(FloatHolder& fha
    , FloatHolder& fhv
    , FloatHolder& fhout)
{/*
  cudaEventRecord(start);
  matxvecDirectProdKernel
      <<<dim3((fhin.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fhin.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fha.nx()
      , fha.ny()
      , fha.rawPtrGPU
      , fhv.rawPtrGPU
      , fhout.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in tranposeFast: %f\n", ms);
  printf("Effective throughput of transposeFast: %f GB/s\n"
      , fhin.totalElements*sizeof(float)/ms/1e6);*/
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
  printf("Elapsed time in matxmatNaive: %f\n", ms);
  printf("Effective throughput of matxmatNaive: %f GB/s\n"
      , fha.totalElements*sizeof(float)/ms/1e6);
}

void KernelLauncher::matxmatFast(FloatHolder& fha
    , FloatHolder& fhb
    , FloatHolder& fhbtranspose
    , FloatHolder& fhout)
{
  cudaEventRecord(start);
  transposeFastKernel
      <<<dim3((fha.nx()+BLOCKDIMX-1)/BLOCKDIMX,(fha.ny()+BLOCKDIMY-1)/BLOCKDIMY), dim3(BLOCKDIMX,BLOCKDIMY)>>>
      (fhb.nx()
      , fhb.ny()
      , fhb.rawPtrGPU
      , fhbtranspose.rawPtrGPU);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Elapsed time in matxmatFast: %f\n", ms);
  printf("Effective throughput of matxmatFast: %f GB/s\n"
      , fha.totalElements*sizeof(float)/ms/1e6);
}

