#include <flash_runtime/flashrt.h>
#include <ranges>

std::shared_ptr<flash_rt> flash_rt::_global_ptr;


std::shared_ptr<flash_rt> flash_rt::get_runtime( std::string runtime_lookup )
{

  if( _global_ptr )
    return _global_ptr;
  else
    return _global_ptr = std::shared_ptr<flash_rt>( new flash_rt() );

}

flash_rt::flash_rt()
{
  std::cout << "Ctor'ing flash_rt...." << std::endl;
}

status flash_rt::execute(std::string kernel_name, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "Executing from RuntimeImpl..." << std::endl;
  return {};
}

status flash_rt::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                        std::string kernel_names[], std::optional<std::string> inputs[] ) 
{
  
 for (int i : std::views::iota((size_t) 1, num_kernels))
 {

 }

 return {}; 
}
