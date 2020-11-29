#include <stdio.h>

__global__ void elmatmult_generic( float * a, float * b, float * c)
{
  
  printf("Testing from elmatmult_generic [%d,%d,%d]\n",threadIdx.x, threadIdx.y, threadIdx.z);
  printf("--Testing from elmatmult_generic %f * %f\n", a[threadIdx.x], b[threadIdx.x] );
  c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
  

  return;
}

__global__ void elmatdiv_generic( float * a, float * b, float * c)
{
  printf("Testing from elmatdiv_generic [%d,%d,%d]\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("--Testing from elmatdiv_generic %f / %f \n", a[threadIdx.x], b[threadIdx.x]);
  c[threadIdx.x] = a[threadIdx.x] / b[threadIdx.x];

  return;
}

