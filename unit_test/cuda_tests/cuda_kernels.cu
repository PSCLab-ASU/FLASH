#include <stdio.h>

__global__ void elmatmult_generic( float * a, float * b, float * c)
{
  printf("Testing from elmatmult_generic\n");
  return;
}

__global__ void elmatdiv_generic( float * a, float * b, float * c)
{

  printf("Testing from elmatdiv_generic\n");
  return;
}

