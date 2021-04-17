#include <iostream>
#include <vector>
#include <flash.h>

//Adapted for FLASH from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/N-BodyMethods/Nbody
                                                                                       //in   //in    //inout   //inout  //inout
using PARTICLE_K = KernelDefinition<2, "particle_init",  kernel_t::EXT_BIN, Sortby<2>, ulong, float*, float**,  float**, float** >; 


struct Particles
{

  //the can be normal std::vectors, but
  //for optimization purposes flash memory can be
  //used as opaque handles to device memory
  //and accessed in the host lazily
  flash_memory<float[3]> pos;
  flash_memory<float[3]> vels;
  flash_memory<float[3]> accs;
  flash_memory<float>    mass;
  float energy;

  size_t n_particles;
 
  Particles(size_t );
  

};

Particles::Particles(size_t n_parts) 
: n_particles( n_parts ), pos(n_parts), 
  vels(n_parts), accs(n_parts), mass(n_parts)
{
  //constructing a runtimei object without a kernels defintion and a kernel impl
  //means that FLASH will leave it up to the backend to find a match
  //from previously linked impls and kernels
  //since PARTICLE_K was registered by the RuntimeObj in main the backend will have
  //a cahced version of PartivleInit
  //since the sumbission kernel description (parameter 1) doesn't override a kernel mehtod
  //it by default uses the one definied in the using statement "Particleit"
  //since no runtime was passed it will choose the runtime based on 
  //a) where the bulk of the input presides, moving the device buffer accordingly
  //b) where there is a valid kernel to execute
  //The sort_by is ignored due to the fact that there is no second dimension in this 
  //launch just a single dimention
  RuntimeObj ocrt;
  ocrt.submit(PARTICLE_K{}, n_particles, mass, pos, vels, accs ).exec( n_particles );

}


int main(int argc, const char * argv[])
{
   
    size_t n_particles=16000, y_stages=2, time_steps=10;

    RuntimeObj ocrt(flash_rt::get_runtime("NVIDIA_GPU") , PARTICLE_K{ argv[0] } );

    //Intitializing is also acclerated by accelerators
    Particles ps(n_particles);

    //calling process_particles method
    //where the x dimension is the number of particles
    //two steps in the 'y' dimension with an implicit barrier
    //and those repeated 10 times
    //flash memory is used to bypass the host write back from the initalization stage
    ocrt.submit(PARTICLE_K{"process_particles"}, n_particles, ps.mass, ps.pos, ps.vel, ps.acc, &ps.energy )
        .exec(n_particles, y_stages, time_steps);

    //Read new positionaa
    auto& final_particle_positions = ps.pos;
    for( auto part_pos :  final_particle_positions.data() )
    {
      std::cout << "x = " << part_pos[0] << ", "
                << "y = " << part_pos[1] << ", " 
      std::cout << "z = " << part_pos[2] << std::endl;
 
    }

    std::cout << "Total energy : " << ps.energy << std::endl;


    return 0;
}

