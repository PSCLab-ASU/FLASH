#include <iostream>
#include <vector>
#include <flash.h>
#include <math.h>
#include <cmath>

//Adapted for FLASH from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/N-BodyMethods/Nbody                                  In      In      Inout   Inout  Inout
                                                                               //grid  //posX  //PosY  //RanX   //RanY
using PARTICLE_K  = KernelDefinition<0, "particle_init", kernel_t::EXT_BIN >; 
using PARTICLE_K2 = KernelDefinition<3, "process_particle", kernel_t::EXT_BIN, GroupBy<2>, ulong, size_t, float, uint64_t, 
                                                                            size_t*, float*, float*, float*, float* >; 

struct Particles
{
  //the can be normal std::vectors, but
  //for optimization purposes flash memory can be
  //used as opaque handles to device memory
  //and accessed in the host lazily
  std::vector<float>  posX;
  std::vector<float>  posY;
  std::vector<float>  randX;
  std::vector<float>  randY;
  std::vector<size_t> grid;

  size_t n_particles;
  size_t n_iter;
 
  Particles(size_t, size_t, ulong, ulong );
  

};

Particles::Particles(size_t n_parts, size_t iter, ulong grid_size, ulong planes) 
: n_iter(iter), n_particles( n_parts ), posX(n_parts), posY(n_parts), 
  randX(n_parts+iter), randY(n_parts+iter), grid( std::pow(grid_size,2) * planes  )
{
 
  size_t final_grid_sz = std::pow(grid_size,2) * planes;  
  //constructing a runtimei object without a kernels defintion and a kernel impl
  //means that app will leave it up to FLASH to find a match
  //from previously linked impls and kernels
  //since PARTICLE_K was registered by the RuntimeObj in main the backend will have
  //a cahced version of particle_init
  //since the sumbission kernel description (parameter 1) doesn't override a kernel mehtod
  //it by default uses the one defined in the using statement PARTICLE_K
  //since no runtime was passed it will choose the runtime based on 
  //a) where the bulk of the input presides, moving the device buffer accordingly
  //b) where there is a valid kernel to execute
  //As a shortcut/optimization I used the "options" method as a named ctor for creating a sliding window
  //to mutate the function declaration it creates a kernel_trait modifier.
  //This helps maximize the reuse of the using statement
  //if you anchor a variable with a single input arg, the position is the number
  //else it is the starting position if multiple args are anchored.
  RuntimeObj ocrt;

  ocrt.options(global_options::DEFER_OUTPUT_DEALLOC)
      .submit(PARTICLE_K{}, posX, posY).defer( n_particles )
      .submit(PARTICLE_K{"grid_init"}, grid).defer( final_grid_sz  )
      .submit(PARTICLE_K{"random_init"}, randX, randY).exec( randX.size() ); 
  //The above statement invokes three kernels, particle_init (implicitly), grid_init, and random_init, respecitvely.
  //with work_item configuration of n_particles, final_grid_sz, randX.size()
  //since all these kernels do not have any dependent parameter small graph optimization
  //will allow these kernels to be scheduled and ran in parallel, compact by default (ie on same device)

}


int main(int argc, const char * argv[])
{
    ulong grid_size=22, planes=3;
    size_t n_particles=256, n_iter=512;
    float radius = 0.5f;

    RuntimeObj ocrt(PARTICLE_K{ argv[0] }, PARTICLE_K2{ argv[0] } );
    //RuntimeObj ocrt( PARTICLE_K{ argv[0] } );
    ocrt.options(global_options::COMMIT_IMPLS);

    //Intitializing is also acclerated by accelerators
    Particles ps(n_particles, n_iter, grid_size, planes);

    //calling process_particles method
    //where the x dimension is the number of particles
    //two steps in the 'y' dimension with an implicit barrier
    //and those repeated 10000 times
    //flash memory is used to bypass the host write back from the initalization stage
    //the fifth argument is a shortcut to create a temporary device buffer for the duration of the subaction
    ocrt.submit(PARTICLE_K2{}, grid_size, n_particles, radius, 
                               n_particles*3, ps.grid, ps.posX, 
                               ps.posY, ps.randX, ps.randY ).exec(n_particles, n_iter);
    
    //transfers data from device to host completely
    /*auto& Xs = ps.posX.data();
    auto& Ys = ps.posY.data();

    
    std::cout << "{ " << Xs[0] << ", " << Ys[0] << " }, ";     
    for( size_t i=1; i < ps.posX.size()-1; i++)
      std::cout << "{ " << Xs[i] << ", " << Ys[i] << " }, ";     
    std::cout << "{ " << Xs[ps.size()-1] << ", " << Ys[ps.size()-1] << " }";     
    */
    return 0;
}

