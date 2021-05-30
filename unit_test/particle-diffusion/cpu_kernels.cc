#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <cmath>
#include <atomic>
#include <cstdlib>

//Adapted from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/StructuredGrids/particle-diffusion/src

extern size_t get_indices( int );
extern void early_terminate();

void particle_init(size_t *, float * posX, float * posY, float *, float *, size_t **)
{
  size_t x = get_indices(0);
  posX[x] = 0;

}

void grid_init( size_t * grid, float *, float *, float *, float *, size_t **)
{
  size_t x = get_indices(0);
  grid[x] = 0;
}

void random_init( size_t *, float *, float *, float * randX, float * randY, size_t **)
{
  size_t x = get_indices(0);
  randX[x] = randY[x] = 0;
}


void process_particles( unsigned long grid_size, size_t n_particles, float radius,
                        unsigned int * prev_known_cell_coordinate_XY,
                        size_t * grid_a, float * particle_X_a, float * particle_Y_a, 
                        float * random_X_a, float * random_Y_a, size_t ** ttable)
{
  const size_t gs2 = grid_size * grid_size;
  size_t p    = get_indices(0);
  size_t iter = ttable[p][1];

  auto prev_known_cell_coordinate_X = prev_known_cell_coordinate_XY[3*p];
  auto prev_known_cell_coordinate_Y = prev_known_cell_coordinate_XY[3*p+1];
  auto inside_cell                  = prev_known_cell_coordinate_XY[3*p+2];

  //std::cout << "3.PRocessing particles... : " << p << "," << iter << "," << n_particles << std::endl;
  // Set the displacements to the random numbers 
  float displacement_X = random_X_a[iter * n_particles + p];
  float displacement_Y = random_Y_a[iter * n_particles + p];
  // Displace particles
  particle_X_a[p] += displacement_X;
  particle_Y_a[p] += displacement_Y;
  // Compute distances from particle position to grid point i.e.,
  // the particle's distance from center of cell. Subtract the 
  // integer value from floating point value to get just the
  // decimal portion. Use this value to later determine if the
  // particle is inside or outside of the cell
  float dX = abs(particle_X_a[p] - round(particle_X_a[p]));
  float dY = abs(particle_Y_a[p] - round(particle_Y_a[p]));

  int iX = floor(particle_X_a[p] + 0.5);
  int iY = floor(particle_Y_a[p] + 0.5);
  
  // Atomic operations flags
  bool increment_C1 = false;
  bool increment_C2 = false;
  bool increment_C3 = false;
  bool decrement_C2_for_previous_cell = false;
  bool update_coordinates = false;

  // Check if particle's grid indices are still inside computation grid
  if ((iX < grid_size) && (iY < grid_size) && (iX >= 0) && (iY >= 0)) {
    // Compare the radius to particle's distance from center of cell
    if (radius >= sqrt(dX * dX + dY * dY)) {
      // Satisfies counter 1 requirement for cases 1, 3, 4
      increment_C1 = true;
      // Case 1 
      if (!inside_cell) {
        increment_C2 = true;
        increment_C3 = true;
        inside_cell = 1;
        update_coordinates = true;
      }
      // Case 3
      else if (prev_known_cell_coordinate_X != iX ||
               prev_known_cell_coordinate_Y != iY) {
        increment_C2 = true;
        increment_C3 = true;
        update_coordinates = true;
        decrement_C2_for_previous_cell = true;
      }
    }
    // Case 2a --Particle remained inside grid and moved outside cell
    else if (inside_cell) {
      inside_cell = 0;
      decrement_C2_for_previous_cell = true;
    }

  }
  else if( inside_cell ){
    inside_cell = 0;
    decrement_C2_for_previous_cell = true;
  }

  // Index variable for 3rd dimension of grid
  size_t layer;
  
  // Current and previous cell coordinates 
  size_t curr_coordinates = iX + iY * grid_size;
  size_t prev_coordinates = prev_known_cell_coordinate_X +
                            prev_known_cell_coordinate_Y * grid_size;

  // Counter 2 layer of the grid (1 * grid_size * grid_size)
  layer = gs2;
  if (decrement_C2_for_previous_cell)
    std::atomic_ref(grid_a[prev_coordinates + layer]).fetch_sub(1);
    //atomic_fetch_sub<size_t>(grid_a[prev_coordinates + layer], 1);

  if (update_coordinates) {
    prev_known_cell_coordinate_X = iX;
    prev_known_cell_coordinate_Y = iY;
  }

  // Counter 1 layer of the grid (0 * grid_size * grid_size)
  layer = 0;
  if (increment_C1)
    std::atomic_ref(grid_a[curr_coordinates + layer]).fetch_add(1);
    //atomic_fetch_add<size_t>(grid_a[curr_coordinates + layer], 1);

  // Counter 2 layer of the grid (1 * grid_size * grid_size)
  layer = gs2;
  if (increment_C2)
    std::atomic_ref(grid_a[curr_coordinates + layer]).fetch_add(1);
    //atomic_fetch_add<size_t>(grid_a[curr_coordinates + layer], 1);

  // Counter 3 layer of the grid (2 * grid_size * grid_size)
  layer = gs2 + gs2;
  if (increment_C3)
    std::atomic_ref(grid_a[curr_coordinates + layer]).fetch_add(1);
    //atomic_fetch_add<size_t>(grid_a[curr_coordinates + layer], 1); 
   
}

