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
#include "elf_ext.h"

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

  CUdeviceptr allocate_dbuffer( size_t bytes )
  {
    CUdeviceptr dptr;

    set_context_to_current();

    cuMemAlloc ( &dptr, bytes );

    return dptr;
  }
 
  void deallocate_dbuffer( CUdeviceptr& dptr )
  {
    cuMemFree( dptr );
    dptr = (CUdeviceptr)0;
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
  std::string id;
  std::string func_name;
  std::string mangled_func_name;
  CUfunction func;

};

struct cuda_kernel
{ 
  //function ID
  std::string cuda_function_id;
  //functiona informaiton
  std::string cuda_function_key;

  std::string cuda_module_key;
  //context information
  std::string cuda_context_key;

  //module information
  size_t module_key;

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

  std::string get_current_ctx_key( )
  {
    CUcontext cuCtx;
    CUdevice cuDevice;
    //get context
    cuCtxGetCurrent( &cuCtx );
    cuCtxGetDevice( &cuDevice);
    
    //find context key
    auto ctx_key = std::ranges::find_if( contexts, [&](auto entry)
    {
      return ( (entry.cuContext == cuCtx) && (entry.cuDevice == cuDevice) );

    } );
    
    return ctx_key->id;

  }

  std::optional<CUfunction> get_function( std::string fid )
  {
    auto entry = std::ranges::find( functions, fid, &cuda_function::id);
    if( entry != functions.end() )
    { 
      std::cout << "Successfully found " << fid << std::endl;
      return entry->func;
    }
    else return {};
  }

  void set_active_context( std::string ctx_key )
  {
    auto ctx = std::ranges::find( contexts, ctx_key, &cuda_context::id );
    if( ctx != contexts.end() )
    {
      ctx->set_context_to_current(); 
    }
    else std::cout << "Could not find ctx to set active" << std::endl;
  }

};

struct kernel_library
{

  void push_back( cuda_kernel cuKern) 
  {
    kernels.push_back( cuKern );
  }

  std::optional<CUfunction>  
  get_cufunction_for_current_ctx( std::string function_name, std::optional<std::string> location = {} )
  {
    auto ctx_key = _kernel_comps.get_current_ctx_key();
    
    auto kernel = std::ranges::find_if( kernels, [&] (auto entry)
    {
      if( location )
        return (entry.cuda_context_key  == ctx_key) &&
               (entry.cuda_module_key   == location.value() ) && 
	       (entry.cuda_function_key == function_name);
      else
        return (entry.cuda_context_key  == ctx_key) &&
	       (entry.cuda_function_key == function_name);


    } );

    auto kfunc = _kernel_comps.get_function( kernel->cuda_function_id );

    return kfunc;

  }

  kernel_components& _kernel_comps;

  //kernels
  std::vector<cuda_kernel> kernels;

};

struct pending_job_t
{
  uint nInputs;
  std::string cuCtx_key;
  std::vector<te_variable> kernel_args;
  std::vector<CUdeviceptr> device_buffers; 

  pending_job_t( std::string ctx_key, std::vector<te_variable> host_buffs, 
		 std::vector<CUdeviceptr> dev_buffs, uint num_inputs)
  {
    cuCtx_key      = ctx_key;
    kernel_args    = host_buffs;
    device_buffers = dev_buffs;
    nInputs        = num_inputs;
  }

  ~pending_job_t()
  {
    std::ranges::for_each(device_buffers, [](auto& dev_buffer )
    {
      cuMemFree( dev_buffer );
    } );
  }

};


//think about creating meta data for the memory
struct mem_detail
{
  struct mem_state{
    std::string tid;
    DIRECTION dir; 
    std::optional<cuda_context> ctx;
    std::optional<CUdeviceptr> dptr;
  };

  mem_detail( void * data, size_t size, bool flash_mem = false) 
  : host_data(data), sz(size), _is_flash(flash_mem) {}

  auto add_memstate( std::string tid, cuda_context ctx, DIRECTION dir )
  {
    auto ms = mem_state{ tid, dir, ctx, {} };
    return _mstates.emplace_back( ms );
  }

  size_t size() { return sz; }
  bool is_flash() { return _is_flash; } 
  void allocate( std::string, cuda_context, DIRECTION, bool init = true );
  void deallocate( std::string, bool transfer_back = false ); 
  void transfer( std::string , cuda_context, DIRECTION , bool );
  void host_to_device( CUdeviceptr );
  void device_to_host( CUdeviceptr );
  void device_to_device(CUcontext, CUdeviceptr, CUcontext, CUdeviceptr );

  void transfer_lastDtoH()
  {
    device_to_host( _mstates.back().dptr.value() );
  }

  void deallocate_all(bool);
 
  std::string last_tid()
  {
    return _mstates.back().tid;
  }

  bool tid_exists( std::string tid )
  {
    return std::ranges::any_of( _mstates, unary_equals{tid}, 
                                &mem_state::tid );
  }

  auto get_mstate( std::string tid)
  {
    return std::ranges::find( _mstates, tid, &mem_state::tid );
  }

  //flash mem ID
  std::vector<mem_state> _mstates;

  size_t sz;
  void * host_data;
  bool _is_flash;
};

class flash_cuda : public IFlashableRuntime
{

  public:

    status register_kernels( const std::vector<kernel_desc> &,
                             std::vector<bool>& ) final;

    status execute( runtime_vars, uint, std::vector<te_variable>, std::vector<size_t> ) final;  

    status wait( ulong ) final;
 
    status allocate_buffer( te_variable&, bool& ) final;

    status deallocate_buffer( std::string, bool&) final;
 
    status deallocate_buffers( std::string )      final;

    status transfer_buffer( std::string, void *)  final;

    static FlashableRuntimeMeta<IFlashableRuntime> get_runtime();

    static std::shared_ptr<flash_cuda> get_singleton();

    static std::string get_factory_name() { return "NVIDIA_GPU"; }

    static std::string get_factory_desc() { return "This runtime supports NVIDIA CUDA"; }


  private:

    flash_cuda();

    void _retain_cuda_contexts();

    void _reset_cuda_ctx();

    void _set_least_active_gpu_ctx();

    void _add_cuda_modules(std::string);

    void _add_cuda_function( std::string );

    void _process_external_binary( const kernel_desc & );

    int _find_cubin_offset( void *, size_t, std::string, size_t *, std::string * );

    kernel_components  _kernel_comps;

    kernel_library _kernel_lib;

    std::map<ulong, pending_job_t> _pending_jobs;

    std::map<std::string, ulong> _job_count_per_ctx;

    std::map<std::string, mem_detail> mem_registry;

    static  std::shared_ptr<flash_cuda> _global_ptr; 

    static bool _registered;

};



