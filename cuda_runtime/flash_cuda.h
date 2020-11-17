#include <cuda.h>
#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <variant>
#include <map>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>
#include <boost/align/aligned_allocator.hpp>
#include <regex>
#include <ranges>

#include "elf.c"

#pragma once


struct cuda_context
{
  std::string id;
  CUdevice cuDevice;
  CUcontext cuContext;
 
  std::string get_id()
  {
    return id;
  }

  int set_context_to_current()
  {
    return cuCtxSetCurrent (cuContext );   
  }

};

struct cuda_module
{	
  std::string location;

  //handle to the implementation
  FILE * file;

  size_t file_size;

  //base address to mmap file
  void * data;

  //file distriptor
  int fd;

  size_t fcubin_base_addr;

  size_t fcubin_size;

  //cubin_offset module
  //ONLY set when functions exists
  std::map<size_t, CUmodule> mods;

  std::string get_id()
  {
    return location;
  }

  template<typename T = void>
  T * get_fbin_data()
  {
    return static_cast<T *> (&( (unsigned char *) data)[fcubin_base_addr] );
  } 

  size_t get_fbin_size()
  {
    return fcubin_size;
  }

  void set_module( size_t offset, CUmodule cuMod)
  {
    mods[offset] = cuMod;
  }

  std::optional<CUmodule> get_module( size_t offset )
  {
    if( mods.find( offset ) != mods.end() )
    {
      return mods.at(offset);
    } 
    else return {};
    
  }

  bool test_and_add( size_t offset )
  {
    CUmodule cuModule;
    int err = cuModuleLoadFatBinary(&cuModule, 
		                    &get_fbin_data<unsigned char>()[offset] );
    if( err == CUDA_SUCCESS )
    {
      set_module(offset, cuModule);

      return true;
    }

    return false;
  }

};

struct cuda_function
{
  std::string func_name;
  std::string mangled_func_name;
  CUfunction func;

};

struct cuda_kernel
{ 
  //functiona informaiton
  std::string cuda_function_key;

  std::string cuda_module_key;
  //context information
  std::string cuda_context_key;

  //module information
  size_t module_key;

};

struct kernel_library
{
  //kernels
  std::vector<cuda_kernel> kernels;

  void push_back( cuda_kernel cuKern) 
  {
    kernels.push_back( cuKern );
  }

};

struct kernel_components
{
  //cuda context per device
  std::vector<cuda_context> contexts;

  //location and module 
  std::vector<cuda_module> modules;
	  
  //function_name and function
  std::vector<cuda_function> functions;

  void push_back( std::variant<cuda_context, cuda_module, cuda_function>&& entry)
  {
    std::visit([&]( auto in)
    {
      if constexpr ( std::is_same_v<cuda_context, decltype(in)> )
        contexts.push_back( std::forward<decltype(in)>( in ) );  
      else if constexpr( std::is_same_v<cuda_module, decltype(in)> )
        modules.push_back( std::forward<decltype(in)>( in ) );  
      else  
        functions.push_back( std::forward<decltype(in)>( in ) );  
        
    }, std::forward<decltype(entry)>( entry ) );
  }

  bool module_exists( std::string location )
  {
    return std::ranges::any_of(modules, 
		               [&](std::string loc){ return loc == location; } , 
			       &cuda_module::location);
  }

};


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

    void _retain_cuda_contexts();

    void _reset_cuda_ctx();

    void _add_cuda_modules(std::string);

    void _add_cuda_function( std::string );

    void _process_external_binary( const kernel_desc & );

    int _find_cubin_offset( void *, size_t, std::string, size_t *, std::string * );

    kernel_components  _kernel_comps;

    kernel_library _kernel_lib;

    std::map<ulong, pending_job_t> _pending_jobs;

    static  std::shared_ptr<flash_cuda> _global_ptr; 

    static bool _registered;

};



