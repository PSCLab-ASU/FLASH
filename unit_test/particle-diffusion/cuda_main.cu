#include <iostream>
#include <cuda.h>
#include <vector>
#include <queue>
#include <thread>
#include <map>
#include <mutex>
#include <array>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <math.h>
#include <cuda.h>

//kernels
void __global__ particle_init(size_t *, float * posX, float * posY, float *, float *, size_t **);
void __global__ grid_init( size_t * grid, float *, float *, float *, float *, size_t **);
void __global__ random_init( size_t *, float *, float *, float * randX, float * randY, size_t **);
void __global__ process_particles( unsigned long grid_size, size_t n_particles, float radius,
                                   unsigned int * prev_known_cell_coordinate_XY,
                                   size_t * grid_a, float * particle_X_a, float * particle_Y_a,
                                   float * random_X_a, float * random_Y_a, size_t ** ttable);



int main(int argc, char * argv[] )
{
  const size_t grid_size=22, planes=3, n_particles=256, n_iter=10000;
  const size_t grid_sz = std::pow(grid_size,2) * planes;
  const size_t nmove = n_particles * n_iter;
  size_t n = n_particles;
  float radius = 0.5f;

  ////////////////////////////////////////////////////////////////////////////////////////

  std::vector<size_t> grid(grid_sz, 0);
  std::vector<size_t> task_table(nmove, 0);
  std::vector<unsigned int> prevXY( 3*n, 0);
  std::vector<float> posX(n, 0), posY(n, 0), 
                     randX(nmove, 0), randY(nmove, 0);

  size_t * d_grid, * d_task_table;
  unsigned int * d_prevXY;
  float * d_posX, * d_posY, * d_randX, * d_randY;

  cudaMalloc(&d_grid, grid.size()*sizeof(decltype(grid)::value_type) );
  cudaMalloc(&d_task_table, task_table.size()*sizeof(decltype(task_table)::value_type) );
  cudaMalloc(&d_prevXY, prevXY.size()*sizeof(decltype(prevXY)::value_type) );
  cudaMalloc(&d_posX, posX.size()*sizeof(decltype(posX)::value_type) );
  cudaMalloc(&d_posY, posY.size()*sizeof(decltype(posY)::value_type) );
  cudaMalloc(&d_randX, randX.size()*sizeof(decltype(randX)::value_type) );
  cudaMalloc(&d_randY, randY.size()*sizeof(decltype(randY)::value_type) );

  for(size_t i=0; i < task_table.size(); i = i + 2)
  {
    task_table[i] = i % n;
    task_table[i + 1] = (size_t) ( (i / n) % n_iter );
    //std::cout << "{" << task_table[i][0] << "," <<
    //                    task_table[i][1] <<  "}" << std::endl;
  }
  
  cudaMemcpy(d_task_table, task_table.data(), 
  	     task_table.size()*sizeof(decltype(task_table)::value_type), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();        // Get error code
  if ( err != cudaSuccess )
  {
    printf("PI CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  // record start time
  auto start = std::chrono::system_clock::now();

  particle_init<<<4, 64>>>( d_grid, d_posX, d_posY, d_randX, d_randY, (size_t **) d_task_table ); 

  cudaDeviceSynchronize();

  err = cudaGetLastError();        // Get error code
  if ( err != cudaSuccess )
  {
    printf("PI CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  
  grid_init<<<4, 363>>>( d_grid, nullptr, nullptr, nullptr, nullptr, nullptr ); 

  cudaDeviceSynchronize();

  err = cudaGetLastError();        // Get error code
  if ( err != cudaSuccess )
  {
    printf("GI CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
 
  random_init<<<512, nmove/512>>>( nullptr, nullptr, nullptr, d_randX, d_randY, nullptr ); 

  cudaDeviceSynchronize();

  if ( err != cudaSuccess )
  {
    printf("RI CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  auto d_tt = d_task_table;
  ///////////////////////////////////////////////
  for(int k =0; k < n_iter; k++)
  {
    process_particles<<<1, n_particles>>>(grid_size, n_particles, radius, d_prevXY, 
                                          d_grid, d_posX, d_posY, 
                                          d_randX, d_randY, (size_t **) d_tt ); 

    d_tt += n_particles;
    cudaDeviceSynchronize();
  }
  


  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>( elapsed_seconds );
  std::cout << "\nTotal Time : " << time_ms.count() << "\n\n";

  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////

  return 0;
}
