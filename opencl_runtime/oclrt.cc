#include "opencl_runtime/oclrt.h"
#include <ranges>

std::shared_ptr<ocl_runtime> ocl_runtime::_global_ptr;

std::shared_ptr<ocl_runtime> ocl_runtime::get_runtime()
{

  if( _global_ptr )
    return _global_ptr;
  else
    return _global_ptr = std::shared_ptr<ocl_runtime>( new ocl_runtime() );

}

ocl_runtime::ocl_runtime()
{
  std::cout << "Ctor'ing oclrt_runtime...." << std::endl;
}

status ocl_runtime::execute(std::string kernel_name, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "Executing from RuntimeImpl..." << std::endl;
  return {};
}

status ocl_runtime::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                      std::string kernel_names[], std::optional<std::string> inputs[] ) 
{
  
 for (int i : std::views::iota((size_t) 1, num_kernels))
 {

 }

 return {}; 
}
