#pragma once
#define datatype unsigned int
#define MATDIMX 64
#define MATDIMY 64
//For 1D kernels
#define BLOCKDIM 1024
// For 2D kernels, use 32x32 thread blocks = BLOCKDIM blocks/thread   
// This is the maximum allowable number on my machine, and         
// suffices for coalesced gmem reads.                             
#define BLOCKDIMX 32
#define BLOCKDIMY 32
#define SCANBLOCKDIM 128
