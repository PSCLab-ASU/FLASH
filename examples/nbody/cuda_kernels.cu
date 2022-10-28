#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_kernel.h>
#include <curand_kernel.h>
#include <curand_kernel.h>

void __device__ _update_particle_accel(float *, float, float * , float&, float&, float& );
void __device__ _calc_energy(  float, float *, float *, float *, float& );

__global__ void particle_init( size_t  n_particles, 
		               float * mass, float * positions, 
			       float * velocities, float * accelerations)
{
  size_t p = 3 * (blockIdx.x*blockDim.x + threadIdx.x);
  float * _pos =  positions; 
  float * _vel =  velocities;
  float * _acc =  positions;
 
  curandStatePhilox4_32_10_t state;
  curand_init(1234, 0, p, &state);
  //////////////////////////////////////////////////////                                                     
  mass[p] = n_particles * curand_uniform(&state); 
  _pos[p] = curand_uniform(&state); 
  _vel[p] = curand_uniform(&state) * ( 2.99999) - 1; 
  _acc[p] = 0;
  //////////////////////////////////////////////////////
  _pos[p + 1] = curand_uniform(&state);
  _vel[p + 1] = curand_uniform(&state) * ( 2.99999) - 1; 
  _acc[p + 1] = 0;
  //////////////////////////////////////////////////////
  _pos[p + 2] = curand_uniform(&state);
  _vel[p + 2] = curand_uniform(&state) * ( 2.99999) - 1; 
  _acc[p + 2] = 0; 
  
}

__global__ void process_particles( unsigned long num_particles, 
                                   float * mass, float * positions, 
				   float * velocities, float * accelerations, 
                                   float * energy, size_t ** ttable)
{
  
  size_t particle_idx  = 3 * (blockIdx.x*blockDim.x + threadIdx.x);
  float * _pos = (float *)  positions; 
  float * _vel = (float *)  velocities;
  float * _acc = (float *)  positions;
  size_t * _tt = (size_t *) ttable;

  int adj_stage_idx = _tt[particle_idx + 1];

  if( adj_stage_idx == 0 )
  {
     float accelx=_acc[particle_idx], 
           accely=_acc[particle_idx + 1], 
           accelz=_acc[particle_idx + 2];

     for( int i=0; i < num_particles; i++)
       _update_particle_accel(&_pos[particle_idx], mass[i], &_pos[3*i], 
                              accelx, accely, accelz);

     _pos[particle_idx] = accelx;
     _pos[particle_idx + 1] = accely;
     _pos[particle_idx + 2] = accelz;
  }
  else
  {
    if( particle_idx == 0 )
    {
      float energy_;

      for( int i=0; i < num_particles; i++)
        _calc_energy(mass[i], &_pos[3*i], &_vel[3*i], &_acc[3*i], energy_ ); 

      *energy = energy_;

    }
  }  
  
}

__device__ void _update_particle_accel(float * cur_pos, float mass, float * pos, 
                                       float& accelx, float& accely, float& accelz )
{
  float dx, dy, dz, dist_sqr, dist_inv;
  const float kSofteningSquared = 1e-3f;
  const float kG = 6.67259e-11f;
  dist_inv = dist_sqr = 0.0f;
      
  dx = pos[0] - cur_pos[0];  // 1flop
  dy = pos[1] - cur_pos[1];  // 1flo
  dz = pos[2] - cur_pos[2];  // 1flop
    
  dist_sqr = dx * dx + dy * dy + dz * dz + kSofteningSquared;  // 6flops
  dist_inv = 1.0f / sqrtf(dist_sqr);

  accelx += dx * kG * mass * dist_inv * dist_inv * dist_inv;  // 6flops
  accely += dy * kG * mass * dist_inv * dist_inv * dist_inv;  // 6flops
  accelz += dz * kG * mass * dist_inv * dist_inv * dist_inv;  // 6flops

}

__device__ void  _calc_energy(  float mass, float * pos, float * vel, float * accel, float& energy)
{
  float dt    = 0.1;

  vel[0] = accel[0] * dt;
  vel[1] = accel[1] * dt;
  vel[2] = accel[2] * dt;
  
  pos[0] = vel[0] * dt;
  pos[1] = vel[1] * dt;
  pos[2] = vel[2] * dt;

  accel[0] = 0.f;
  accel[1] = 0.f;
  accel[2] = 0.f;

  energy += (mass * (vel[0] * vel[0] + 
                     vel[1] * vel[1] + 
                     vel[2] * vel[2] ) );  // 7flops

}
