#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>

struct ocl_kernel_details_t
{


};

struct ocl_context_t
{
  ocl_kernel_details_t _kernel_details;
};


class ocl_runtime : public IFlashableRuntime
{

  public:

    status register_kernels( size_t, kernel_t [], std::string [], std::optional<std::string> [] );

    status execute( std::string, uint, std::vector<te_variable>, std::vector<te_variable> ); 

    static FlashableRuntimeMeta<IFlashableRuntime> get_runtime();

    static std::shared_ptr<ocl_runtime> get_singleton();

    static std::string get_factory_name() { return "INTEL_FPGA"; }

    static std::string get_factory_desc() { return "This runtime support opencl rutimes for Altera FPGA"; }


  private:

    ocl_runtime();

    std::vector<ocl_kernel_details_t> _kernels;

    std::vector<ocl_context_t> _devices;

    static  std::shared_ptr<ocl_runtime> _global_ptr; 

    static bool _registered;

};



