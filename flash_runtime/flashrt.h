#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <memory>



struct kernel_details
{

};

struct device_state
{

  kernel_details _kernel_details;
};


class flash_rt
{

  public:

    static std::shared_ptr<flash_rt> get_runtime( std::string );
 
    status register_kernels( size_t, kernel_t [], std::string [], std::optional<std::string> [] );

    status execute( std::string, uint, std::vector<te_variable>, std::vector<te_variable> ); 


  private:

    flash_rt();

    std::vector<kernel_details> _kernels;

    std::vector<device_state> _devices;

    static std::shared_ptr<flash_rt> _global_ptr; 
    
};



