#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ 
void attention_1 (
    const int n,
    const int d,
    const float*__restrict__ key, 
    const float*__restrict__ query, 
    float*__restrict__ dot_product, 
    float*__restrict__ exp_sum ) 
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < n) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
    atomicAdd(exp_sum, expf(sum));
  }
}

__global__ 
void attention_2 (
    const int n,
    const float*__restrict__ exp_sum, 
    const float*__restrict__ dot_product, 
    float*__restrict__ score )
{

  int i = blockIdx.x * blockDim.x + threadIdx.x;  
  if (i < n)
    score[i] = expf(dot_product[i]) / exp_sum[0];
}

__global__ 
void attention_3 (
    const int n,
    const int d,
    const float*__restrict__ score, 
    const float*__restrict__ value, 
    float*__restrict__ output ) 
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;  
  if (j < d) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}

