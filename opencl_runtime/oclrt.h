#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>
#include <opencl.h>
#include <boost/align/aligned_allocator.hpp>

#pragma once

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
   //queues per device
   std::vector<cl_command_queue> _queues;
   
};

struct pending_job_t
{
  uint num_inputs;
  std::vector<te_variable> kernel_args;
  std::vector<cl_mem> device_buffers;
  cl_command_queue cq;
  cl_event event;

  
  void set_event( cl_event&& cvt )
  {
    event = cvt;
  }
  //if the dir = TO_HOST
  //src index must be  within num
  template<MEM_MOVE dir = MEM_MOVE::TO_HOST >
  status transfer( uint, uint );

  template<MEM_MOVE dir = MEM_MOVE::TO_HOST >
  status transfer_all(); 

};


class ocl_runtime : public IFlashableRuntime
{

  public:

    status register_kernels( const std::vector<kernel_desc> & ) final;

    status execute( runtime_vars, uint, std::vector<te_variable>, std::vector<size_t> ) final;  

    status wait( ulong ) final;

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

    std::tuple<cl_context, cl_kernel, cl_command_queue, bool, cl_device_id>
    _try_get_exec_parms( std::string );

    ulong _add_to_pending_jobs( uint, std::vector<te_variable>, std::vector<cl_mem>, cl_command_queue );

    std::vector<ocl_context_t>     _contexts;
    std::vector<kernel_desc>       _kernels;
    std::map<cl_device_id, long>   _device_usage_table;
    std::map<ulong, pending_job_t> _pending_jobs;

    static  std::shared_ptr<ocl_runtime> _global_ptr; 

    static bool _registered;

};



