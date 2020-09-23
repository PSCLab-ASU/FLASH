#include "opencl_runtime/oclrt.h"

std::shared_ptr<oclrt_runtime> oclrt_runtime::_global_ptr;

std::shared_ptr<oclrt_runtime> oclrt_runtime::get_runtime()
{

  if( _global_ptr )
    return _global_ptr;
  else
    return _global_ptr = std::shared_ptr<oclrt_runtime>( new oclrt_runtime() );

}

oclrt_runtime::oclrt_runtime()
{
  std::cout << "Ctor'ing oclrt_runtime...." << std::endl;
}

status oclrt_runtime::execute(std::string kernel_name, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "Executing from RuntimeImpl..." << std::endl;
  return {};
}

