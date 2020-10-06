#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <memory>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>

struct kernel_details
{

};

struct device_state
{

  kernel_details _kernel_details;
};


class flash_rt
{

  using FlashableRuntimeInfo = FlashableRuntimeMeta<IFlashableRuntime>;

  public:

    static std::shared_ptr<flash_rt> get_runtime( std::string );
 
    status register_kernels( size_t, kernel_t [], std::string [], std::optional<std::string> [] );

    status execute( runtime_vars, uint, std::vector<te_variable>, std::vector<te_variable> ); 


  private:
  

    flash_rt( std::string );

    std::optional<FlashableRuntimeInfo>  _backend;
    std::shared_ptr<IFlashableRuntime>   _runtime_ptr;

    std::vector<kernel_details> _kernels;

    std::vector<device_state> _devices;

    static std::shared_ptr<flash_rt> _global_ptr; 
    
};



