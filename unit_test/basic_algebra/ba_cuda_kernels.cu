#include <stdio.h>

__global__ void elwise_matmult( int, float * a, float * b, float * c)
{
  
  printf("Testing from CUDA elwise_matmult [%d,%d,%d]\n",threadIdx.x, threadIdx.y, threadIdx.z);
  printf("--Testing from CUDA  elwise_matmult  %f * %f\n", a[threadIdx.x], b[threadIdx.x] );
  //c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
  

  return;
}

__global__ void elmatdiv_generic( float * a, float * b, float * c)
{
  printf("Testing from elmatdiv_generic [%d,%d,%d]\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("--Testing from elmatdiv_generic %f / %f \n", a[threadIdx.x], b[threadIdx.x]);
  c[threadIdx.x] = a[threadIdx.x] / b[threadIdx.x];

  return;
}

