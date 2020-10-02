#include <flash_runtime/flashrt.h>
#include <ranges>

std::shared_ptr<flash_rt> flash_rt::_global_ptr;


std::shared_ptr<flash_rt> flash_rt::get_runtime( std::string runtime_lookup )
{

  if( _global_ptr )
    return _global_ptr;
  else
    return _global_ptr = std::shared_ptr<flash_rt>( new flash_rt( runtime_lookup) );

}

flash_rt::flash_rt( std::string lookup)
: _backend( FlashableRuntimeFactory::Create( lookup ) )
{
  //get pointer to backend runtime
  _runtime_ptr = _backend.value()();
  std::cout << "Ctor'ing flash_rt...." << std::endl;

  std::cout << _backend->get_description() << std::endl; 
  
}

status flash_rt::execute(std::string kernel_name, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  //check if thier is a runtime exists
  if( _runtime_ptr )
  {
    _runtime_ptr->execute(kernel_name, num_of_inputs, kernel_args, exec_parms );  
  }else std::cout << "No runtime available" << std::endl;
  return {};
}

status flash_rt::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                        std::string kernel_names[], std::optional<std::string> inputs[] ) 
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  //check if thier is a runtime exists
  if( _runtime_ptr )
  {
    _runtime_ptr->register_kernels(num_kernels, kernel_types, kernel_names, inputs );  
  }else std::cout << "No runtime available" << std::endl;

 return {}; 
}
