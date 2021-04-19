#include <cuda.h>

//Adapted from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/N-BodyMethods/Nbody

__device__ unsigned long get_global_id()
{
  
  unsigned long  threadsPerBlock  = blockDim.x * blockDim.y;
  unsigned long  threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
  unsigned long  blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
  unsigned long  globalThreadNum  = blockNumInGrid * threadsPerBlock + threadNumInBlock;

  return globalThreadNum;
}

__global__ void init( unsigned long, float * arg, float * arg_next, size_t ** ttable)
{
  int x = threadIdx.x;

  arg[x] = arg_next[x] = 0;
}

__global__ void compute_heat( unsigned long num_p, float * arg, float * arg_next, size_t ** ttable)
{
  size_t global_wid = get_global_id();
  int x = threadIdx.x;
  float C  = 0.5;

  size_t adj_stage_idx = ttable[global_wid][1];

  if( adj_stage_idx == 0 )
  {
    if( x == num_p + 1)
      arr_next[x] = arr[x - 1];
    else
      arr_next[x] = C * (arr[x + 1] - 2 * arr[x] + arr[x - 1]) + arr[x];
  }
  else
  {
    //swap operation
    arg[x] = arg_next[x];
  }

}

