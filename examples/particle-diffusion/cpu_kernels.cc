#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <cmath>


//Adapted from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/StructuredGrids/particle-diffusion/src

extern size_t get_indices( int );
extern void early_terminate();

void particle_init( unsigned long, float *, float * posX, float * posY, float *, float *, size_t **)
{
  size_t x = get_indices(0);
  posX[x] = 0;

}

void grid_init( unsigned long, float * grid, float *, float *, float *, float *, size_t **)
{
  size_t x = get_indices(0);
  grid[x] = 0;
}

void random_init( unsigned long, float *, float *, float *, float * randX, float * randY, size_t **)
{
  size_t x = get_indices(0);
  randX[x] = randY[x] = 0;
}


void process_particles( unsigned long grid_size, 
                        float * grid, float * posX, float * posY, 
                        float * randX, float * randY, size_t ** ttable)
{

}

