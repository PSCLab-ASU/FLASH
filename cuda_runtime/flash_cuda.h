#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>
#include <boost/align/aligned_allocator.hpp>

#pragma once


struct pending_job_t
{
  uint num_inputs;
  std::vector<te_variable> kernel_args;
  
  //if the dir = TO_HOST
  //src index must be  within num
  template<MEM_MOVE dir = MEM_MOVE::TO_HOST >
  status transfer( uint, uint );

  template<MEM_MOVE dir = MEM_MOVE::TO_HOST >
  status transfer_all(); 

};


class flash_cuda : public IFlashableRuntime
{

  public:

    status register_kernels( const std::vector<kernel_desc> & ) final;

    status execute( runtime_vars, uint, std::vector<te_variable>, std::vector<size_t> ) final;  

    status wait( ulong ) final;

    static FlashableRuntimeMeta<IFlashableRuntime> get_runtime();

    static std::shared_ptr<flash_cuda> get_singleton();

    static std::string get_factory_name() { return "NVIDIA_GPU"; }

    static std::string get_factory_desc() { return "This runtime supports NVIDIA CUDA"; }


  private:

    flash_cuda();

    std::map<ulong, pending_job_t> _pending_jobs;

    static  std::shared_ptr<flash_cuda> _global_ptr; 

    static bool _registered;

};



