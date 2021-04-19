#include <opencl.h>


//Adapted from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/StructuredGrids/particle-diffusion/src


__kernel void particle_init( unsigned long, float *, float * posX, float * posY, float *, float *, size_t **)
{
  size_t x = get_local_id(0);
  posX[x] = 0;

}

__kernel void grid_init( unsigned long, float * grid, float *, float *, float *, float *, size_t **)
{
  size_t x = get_local_id(0);
  grid[x] = 0;
}

__kernel void random_init( unsigned long, float *, float *, float *, float * randX, float * randY, size_t **)
{
  size_t x = get_local_id(0);
  randX[x] = randY[x] = 0;
}


__kernel void process_particles( unsigned long grid_size, 
                                 float * grid, float * posX, float * posY, 
                                 float * randX, float * randY, size_t ** ttable)
{

}

