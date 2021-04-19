#include <cuda.h>


//Adapted from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/StructuredGrids/particle-diffusion/src

__global__ void particle_init( unsigned long, float *, float * posX, float * posY, float *, float *, size_t **)
{
  int x = threadIdx.x;
  posX[x] = 0;

}

__global__ void grid_init( unsigned long, float * grid, float *, float *, float *, float *, size_t **)
{
  int x = threadIdx.x;
  grid[x] = 0;
}

__global__ void random_init( unsigned long, float *, float *, float *, float * randX, float * randY, size_t **)
{
  int x = threadIdx.x
  randX[x] = randY[x] = 0;
}


__global__ void process_particles( unsigned long grid_size, 
                                   float * grid, float * posX, float * posY, 
                                   float * randX, float * randY, size_t ** ttable)
{

}

