#include "opencl_runtime/oclrt.h"
#include <iostream>
#include <ranges>


/* Registers the factory with flash factory*/
bool ocl_runtime::_registered = FlashableRuntimeFactory::Register(
                                ocl_runtime::get_factory_name(),
                                ocl_runtime::get_runtime() );

std::shared_ptr<ocl_runtime> ocl_runtime::_global_ptr;


FlashableRuntimeMeta<IFlashableRuntime> ocl_runtime::get_runtime()
{
  //automatic polymorphism to base classa
  FlashableRuntimeMeta<IFlashableRuntime> out{ (std::shared_ptr<IFlashableRuntime> (*)()) get_singleton, 
                                                get_factory_desc() };

  return out;
}


std::shared_ptr<ocl_runtime> ocl_runtime::get_singleton()
{

  if( _global_ptr ) return _global_ptr;
  else return _global_ptr = std::shared_ptr<ocl_runtime>( new ocl_runtime() );

}

ocl_runtime::ocl_runtime()
{
  std::cout << "Ctor'ing oclrt_runtime...." << std::endl;
}

status ocl_runtime::execute(std::string kernel_name, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "Executing from opencl_runtime..." << std::endl;
  return {};
}

status ocl_runtime::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                      std::string kernel_names[], std::optional<std::string> inputs[] ) 
{
  
  std::cout << "calling ocl_runtime::" << __func__<<  std::endl;
 
  return {}; 
}
