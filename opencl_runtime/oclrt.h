#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <flash_interface.h>

struct ocl_kernel_details_t
{


};

struct ocl_context_t
{

  kernel_details _kernel_details;
};






class ocl_runtime : public flash_interface
{

  public:

    static std::shared_ptr<ocl_runtime> get_runtime();
 
    status register_kernels( size_t, kernel_t [], std::string [], std::optional<std::string> [] );

    status execute( std::string, uint, std::vector<te_variable>, std::vector<te_variable> ); 


  private:

    oclrt_runtime();

    std::vector<kernel_details> _kernels;

    std::vector<device_state> _devices;

    static  std::shared_ptr<ocl_runtime> _global_ptr; 

};



