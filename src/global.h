#pragma once
#define datatype unsigned int
#define MATDIM  (1024*1024*128)
#define MATDIMX 128
#define MATDIMY 128
//For 1D kernels
#define BLOCKDIM 1024
// For 2D kernels, use 32x32 thread blocks = BLOCKDIM blocks/thread   
// This is the maximum allowable number on my machine, and         
// suffices for coalesced gmem reads.                             
#define BLOCKDIMX 32
#define BLOCKDIMY 32
#define SCANSECTION 128
#define DIVUP(X,Y) ((X+Y-1)/Y)
#define PADTOSECDIM(n) (SCANSECTION*((n+SCANSECTION-1)/SCANSECTION))
