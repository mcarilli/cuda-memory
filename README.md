cuda-memory
===========

Playing around with GPU memory optimization

Current Kernels
All operate on 2D matrices, set up in memory as pseudo-2D arrays.

copy -- Simple copy as benchmark.

saxpy -- Single-precision ax+y; benchmark.

transposeNaive -- Transposes a matrix out-of-place naively.  One element per thread, all memory accesses global, some not coalesced.

transposeFast -- Transposes a matrix using shared memory tiles as an intermediate step to coalesce gmem access.

transposeFastNoBankConf -- Same as transposeFast, but eliminates bank conflicts in smem tiles.

matxmatNaive -- Naive matrix multiply.  Each thread handles one element of output array; all memory accesses are global, some not coalesced.

matxmatTiles -- Matrix multiply using shared memory tiles to ensure that all gmem accesses are coalesced.  Smem bank conflicts are also eliminated.

reduceY -- Reduces (sums) columns of matrix into top row.
