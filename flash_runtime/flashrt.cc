#include <flash_runtime/flashrt.h>
#include <ranges>
#include <algorithm>


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

status flash_rt::execute(runtime_vars rt_vars,  uint num_of_inputs, 
                         std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  //check if thier is a runtime exists
  if( _runtime_ptr )
  {
    _runtime_ptr->execute(rt_vars, num_of_inputs, kernel_args, exec_parms );  
  }else std::cout << "No runtime available" << std::endl;
  return {};
}

status flash_rt::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                        std::string kernel_names[], std::optional<std::string> inputs[] ) 
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  std::vector<kernel_desc> kernel_inputs;

  auto pack_data = [&](int index)-> kernel_desc
  {
    return kernel_desc{kernel_types[index], kernel_names[index], inputs[index]};
  };

  //check if thier is a runtime existsa
  auto kernels = std::views::iota( (size_t) 0, num_kernels ) | std::views::transform(pack_data);
  
  //pack the inputs into
  std::ranges::for_each(kernels, [&](auto input){ kernel_inputs.push_back(input); } );
  
  if( _runtime_ptr )
  {
    _runtime_ptr->register_kernels( kernel_inputs  );  
  }else std::cout << "No runtime available" << std::endl;

 std::cout << "completed flash_rt::" << __func__ << std::endl;
 return {}; 
}
