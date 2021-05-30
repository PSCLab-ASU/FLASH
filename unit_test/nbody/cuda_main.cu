#include <iostream>
#include <cuda.h>
#include <vector>
#include <queue>
#include <thread>
#include <map>
#include <mutex>
#include <array>
#include <chrono>
#include <cuda.h>

void __global__ particle_init( unsigned long, float * mass, float ** positions, float ** velocities, float ** accelerations, size_t ** ttable);
void __global__ process_particles( unsigned long num_particles,float * mass, float ** positions, 
                                   float ** velocities, float ** accelerations, float * energy, size_t ** ttable);

int main(int argc, char * argv[] )
{
  const size_t n_particles=256*64, y_stages=2, time_steps=10;
  size_t n = n_particles;
  float energy =0; 
  float * d_energy;
  size_t tt_size= n*y_stages*time_steps;

  ////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "Hello World : " <<  std::endl; 
  float  *  d_masses;
  float  *  d_positions; 
  float  *  d_velocities;
  float  *  d_accelerations;
  size_t *  d_task_table;
  std::vector<float>    masses(n);
  float ** positions;
  float ** velocities;
  float ** accelerations;
  //std::vector<size_t *> task_table(n*y_stages*time_steps, 0);
  
  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////

  //allocate buffer
  //for(auto i : std::views::iota((size_t)0, n) )
  
  cudaMalloc((void **) &d_energy,            sizeof(float)  );
  cudaMalloc((void **) &d_masses,          n*sizeof(float)  );
  cudaMalloc((void **) &d_positions,     3*n*sizeof(float*) );
  cudaMalloc((void **) &d_velocities,    3*n*sizeof(float*) );
  cudaMalloc((void **) &d_accelerations, 3*n*sizeof(float*) );
  cudaMalloc((void **) &d_task_table,    3*tt_size*sizeof(size_t) );

  positions     =  (float **) new float[n][3];
  velocities    =  (float **) new float[n][3];
  accelerations =  (float **) new float[n][3];

  size_t * task_table = new size_t[3*tt_size]; 

  for(size_t i=0; i < tt_size; i++ )
  {
   
    task_table[3*i] = i % n;
    task_table[3*i + 1] = (size_t) ( (i / n) % y_stages );
    task_table[3*i + 2] = (size_t) ( (i / (y_stages*n ) % time_steps) );

    /*std::cout << "{" << task_table[i][0] << "," <<
                        task_table[i][1] << "," << 
                        task_table[i][2] << "}" << std::endl;*/
  }

  cudaMemcpy(d_task_table, task_table, 3*tt_size*sizeof(size_t), cudaMemcpyHostToDevice);

  cudaError_t err = cudaGetLastError();        // Get error code

  if ( err != cudaSuccess ){
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  // record start time
  auto start = std::chrono::system_clock::now();

  auto d_tt = d_task_table;
  particle_init<<<256, 64>>>(n_particles, d_masses, (float **) d_positions,
                            (float **) d_velocities, (float **) d_accelerations, 
			    (size_t **) d_task_table );
  
  err = cudaGetLastError();        // Get error code

  if ( err != cudaSuccess ){
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  for(size_t i=0; i < time_steps; i++)
  {
      
    std::cout <<"Time Step : " << i << std::endl;
    process_particles<<<256, 64>>>(n_particles, d_masses, (float **) d_positions,
                                   (float**) d_velocities, (float **) d_accelerations,
                                   d_energy, (size_t **) d_tt );
  
    err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess ){
      printf("1) CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }
    
    d_tt += 3*n_particles;

    process_particles<<<1, 1>>>(n_particles, d_masses, (float **) d_positions,
                                (float **) d_velocities, (float **) d_accelerations,
                                 d_energy, (size_t **) d_tt );

    err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess ){
      printf("2) CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
    }

    d_tt += 3*n_particles;
   
  } 
  cudaDeviceSynchronize();

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>( elapsed_seconds );
  std::cout << "\nTotal Time : " << time_ms.count() << "\n\n";

  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  delete [] positions; 
  delete [] velocities; 
  delete [] accelerations; 
  delete [] task_table;

  return 0;
}
