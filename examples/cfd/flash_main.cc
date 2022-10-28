#include <iostream>
#include <vector>
#include <flash.h>
#include <cfd.h>

using CFD_INIT     = KernelDefinition<2, "init_variables", 
                       kernel_t::EXT_BIN,int, float *, float * >; 

using COMPUTE_SF   = KernelDefinition<3, "compute_step_factor", 
                       kernel_t::EXT_BIN, int, float *, float *, float *>;

using COMPUTE_FLUX = KernelDefinition<9, "compute_flux", 
                       kernel_t::EXT_BIN, int, int *, float *, float *,
		                          float *, float *, Float3 *,
					  Float3 *, Float3 *, Float3 *>;

using TIME_STEP    = KernelDefinition<5, "time_step",
                       kernel_t::EXT_BIN, int, int, float *, float *, float *, float *>;


int main(int argc, const char * argv[])
{
  int nelr = 1024, iters=8;
  flash_memory<float> old_vars(nelr*NVAR), vars(nelr*NVAR), ff_vars(nelr*NVAR), areas(nelr*NVAR);
  flash_memory<float> fluxes(nelr*NVAR), step_factors(nelr), null_vector(0);
  flash_memory<float> flux_contribution_momentum_x(3), flux_contribution_momentum_y(2);
  flash_memory<float> flux_contribution_momentum_z(3), flux_contribution_density_energy(3);
  flash_memory<float> normals(nelr*NDIM*NNB);
  flash_memory<int>   elements_surrounding_elements(nelr*NNB);

  ulong gridx = ((nelr + BLOCK_SIZE_2 - 1)/BLOCK_SIZE_2),
        gridx1=(nelr + BLOCK_SIZE_1 - 1)/BLOCK_SIZE_1, blkx = BLOCK_SIZE_1,
        gridDim3=((nelr + BLOCK_SIZE_3 - 1)/BLOCK_SIZE_3), 
	gridDim4=((nelr + BLOCK_SIZE_4 - 1)/BLOCK_SIZE_4);


  RuntimeObj ocrt( CFD_INIT{argv[0]} );

  ocrt.submit(CFD_INIT{}, nelr, vars, ff_vars).defer(gridx1, 1UL, 1UL, blkx)
      .submit(CFD_INIT{}, nelr, old_vars, ff_vars).defer(gridx1, 1UL, 1UL, blkx)
      .submit(CFD_INIT{}, nelr, fluxes, ff_vars).defer(gridx1, 1UL, 1UL, blkx)
      .submit(CFD_INIT{"init_buffer"}, nelr, step_factors, null_vector ).exec( gridx1, 1UL, 1UL, blkx); 


  auto lv = loop_var(2);
  // for the first iteration we compute the time step
  ocrt.submit(COMPUTE_SF{}, nelr, vars, areas, step_factors)
        .defer(gridx, 1UL, 1UL, BLOCK_SIZE_2)
      .submit(COMPUTE_FLUX{}, nelr, elements_surrounding_elements, normals, vars, ff_vars, fluxes,
	      flux_contribution_density_energy, flux_contribution_momentum_x,
	      flux_contribution_momentum_y, flux_contribution_momentum_y,
	      flux_contribution_momentum_z)
        .defer(gridDim3, 1UL, 1UL, BLOCK_SIZE_3 )
      .submit(TIME_STEP{}, lv, nelr, old_vars, vars, step_factors, fluxes)
	.defer(gridDim4, 1UL, 1UL, BLOCK_SIZE_4)
      .exec( iters, RK ); //default cascade_loop

  return 0;

}

