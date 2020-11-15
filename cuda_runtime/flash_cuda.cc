#include "cuda_runtime/flash_cuda.h"
#include <cuda.h>
#include <iostream>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <climits>



/* Registers the factory with flash factory*/
bool flash_cuda::_registered = FlashableRuntimeFactory::Register(
                                flash_cuda::get_factory_name(),
                                flash_cuda::get_runtime() );

std::shared_ptr<flash_cuda> flash_cuda::_global_ptr;


FlashableRuntimeMeta<IFlashableRuntime> flash_cuda::get_runtime()
{
  //automatic polymorphism to base classa
  FlashableRuntimeMeta<IFlashableRuntime> out{ (std::shared_ptr<IFlashableRuntime> (*)()) get_singleton, 
                                                get_factory_desc() };

  return out;
}


std::shared_ptr<flash_cuda> flash_cuda::get_singleton()
{

  if( _global_ptr ) return _global_ptr;
  else return _global_ptr = std::shared_ptr<flash_cuda>( new flash_cuda() );

}

flash_cuda::flash_cuda()
{
  std::cout << "Ctor'ing flash_cuda...." << std::endl;




}

status flash_cuda::wait( ulong  wid)
{
  //get job from pending jobs  
  
 
  return status{};
}

status flash_cuda::execute(runtime_vars rt_vars, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms)
{
  std::cout << "Executing from opencl_runtime..." << __func__ << std::endl;
  std::cout << "Executing : " << rt_vars.get_lookup() <<" ..."<< std::endl;







  return status{-1};
}

status flash_cuda::register_kernels( const std::vector<kernel_desc>& kds ) 
{
  
  std::cout << "calling flash_cuda::" << __func__<<  std::endl;


  return {}; 
}


template<MEM_MOVE dir>
status pending_job_t::transfer(uint src, uint dst)
{

  if( dir == MEM_MOVE::TO_DEVICE ) 
  {
    //copy data to device buffer
  }
  else
  {
    //copy data to device buffer

  }

  return status{};

}

//does not support READ_WRITE parameters yet
template<MEM_MOVE dir>
status pending_job_t::transfer_all()
{

  if( dir == MEM_MOVE::TO_DEVICE ) 
  {
    for( auto i : std::views::iota((uint)0, num_inputs) ) 
      transfer<dir>(i, i);
  }
  else
  {
    for( auto i : std::views::iota(num_inputs, kernel_args.size() ) ) 
      transfer( i, i );

  }
  return status{};
}

