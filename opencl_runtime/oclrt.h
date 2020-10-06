#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>
#include <opencl.h>

struct ocl_program_t
{
  std::optional<std::string> impl_location;
  std::optional<cl_program> program;
  std::map<std::string, std::optional<cl_kernel> > kernels;
};

struct ocl_context_t
{
   //total context
   cl_context _ctx;
   //some device breakdown
   std::vector<cl_device_id> _dev_ids;
   //  def:kernel_name   programs
   std::vector<ocl_program_t > _programs;
};


class ocl_runtime : public IFlashableRuntime
{

  public:

    status register_kernels( const std::vector<kernel_desc> & ) final;

    status execute( runtime_vars, uint, std::vector<te_variable>, std::vector<te_variable> ) final;  

    static FlashableRuntimeMeta<IFlashableRuntime> get_runtime();

    static std::shared_ptr<ocl_runtime> get_singleton();

    static std::string get_factory_name() { return "INTEL_FPGA"; }

    static std::string get_factory_desc() { return "This runtime support opencl runtimes for Altera FPGA"; }


  private:

    ocl_runtime();

    std::optional<std::vector<cl_device_id> > _get_devices();

    //convert a kernel desciption into reading the aocx file
    std::vector<
      std::tuple<std::string, std::optional<std::string>,
                std::optional<std::string> > >
    _read_kernel_files( const std::vector<kernel_desc>& );

    void _append_programs_kernels( auto ); 

    std::vector<ocl_context_t> _contexts;
    std::vector<kernel_desc>   _kernels;

    static  std::shared_ptr<ocl_runtime> _global_ptr; 

    static bool _registered;

};



