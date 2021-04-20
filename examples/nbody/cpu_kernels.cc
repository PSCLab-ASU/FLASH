#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <cmath>


//Adapted from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/N-BodyMethods/Nbody

extern size_t get_indices( int );
extern void early_terminate();

void _update_particle_accel(float *, float, float * , float&, float&, float& );
void _calc_energy(  float, float *, float *, float *, float& );


void particle_init( unsigned long, float * mass, float ** positions, float ** velocities, float ** accelerations, size_t ** ttable)
{
  std::random_device rd;
  std::mt19937 gen(42);
  float base_unit = 1.0e-3f;

  std::uniform_real_distribution<float> unif_d(0, 1.0);

  size_t x = get_indices( 0 );

  positions[x][0] = unif_d(gen);
  positions[x][1] = unif_d(gen);
  positions[x][2] = unif_d(gen);

  mass[x] = 10 * unif_d(gen);

  unif_d = std::uniform_real_distribution<float>( -1.0, 1.0 );

  velocities[x][0] = unif_d(gen) * base_unit;  
  velocities[x][1] = unif_d(gen) * base_unit;  
  velocities[x][2] = unif_d(gen) * base_unit;  

  accelerations[x][0] = 0.f;
  accelerations[x][1] = 0.f;
  accelerations[x][2] = 0.f;

}

void process_particles( unsigned long num_particles, 
                        float * mass, float ** positions, float ** velocities, float ** accelerations, 
                        float * energy, size_t ** ttable)
{
  size_t work_item_id  = get_workitem_idx();
  size_t particle_idx  = get_indices( 0 );

  size_t adj_stage_idx = ttable[work_item_id][1];

  if( adj_stage_idx == 0 )
  {
     float accelx=accelerations[particle_idx][0], 
           accely=accelerations[particle_idx][1], 
           accelz=accelerations[particle_idx][2];

     #pragma omp parallel for default(shared) reduction(+:accelx, accely, accelz)
     for( int i=0; i < num_particles; i++)
       _update_particle_accel(positions[particle_idx], mass[i], positions[i], 
                              accelx, accely, accelz);

     positions[particle_idx][0] = accelx;
     positions[particle_idx][1] = accely;
     positions[particle_idx][2] = accelz;
  }
  else
  {
    if( particle_idx == 0 )
    {

      float energy_;
      #pragma omp parallel for default(shared) reduction(+:energy_)
      for( int i=0; i < num_particles; i++)
      _calc_energy(mass[i], positions[i], velocities[i], accelerations[i], energy_ ); 
      *energy = energy_;

      early_terminate();
    }
  }

}

void _update_particle_accel(float * cur_pos, float mass, float * pos, 
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

  accelx += dx * kG * mass * std::pow(dist_inv,3);  // 6flops
  accely += dy * kG * mass * std::pow(dist_inv,3);  // 6flops
  accelz += dz * kG * mass * std::pow(dist_inv,3);  // 6flops

}

void _calc_energy(  float mass, float * pos, float * vel, float * accel, float& energy)
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

  energy += (mass * (std::pow(vel[0], 2) + 
                     std::pow(vel[1], 2) + 
                     std::pow(vel[2], 2)) );  // 7flops

}
