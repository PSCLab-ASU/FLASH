#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <variant>
#include <utils/common.h>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>

#pragma once

struct options
{

};

//used to keep track of submission parameters
//to propagate throught the builder
struct te_submit_params
{
  //backrop
  std::vector<te_variable> _params;

  //forward prop
  std::vector<size_t> _sizes;
  std::optional<size_t> _dependency;
  
};


//used to propagate runtime attributes through
//the builder
struct te_runtime_params
{
  //backprop
  std::vector<options> _options;

};


struct prop_vehicle
{
  using dispatch_set = std::pair<te_submit_params,
                                 te_runtime_params>;

  std::vector<dispatch_set> _submissions;

};


struct subaction
{
  ulong subaction_id;
  uint num_inputs;
  runtime_vars rt_vars;
  std::vector<te_variable> kernel_args;
  std::vector<size_t> exec_parms;
  options opt;

  auto input_vars( ) 
  { 
    return std::make_tuple(num_inputs, rt_vars, kernel_args, exec_parms );
  };

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
class flash_rt;

class runtimes_resource_tracker
{

  struct summary_flash_mem{
    std::string id;
    uint type_size;
    size_t vec_size;
    void * base_addr;
    std::string tid;
  };

  struct summary_kernel{
    std::string kernel_name;
    std::string kernel_location;
  };

  typedef std::variant<summary_kernel, summary_flash_mem> resource;
 
  public:
  
    using shared_flash_runtime = std::shared_ptr<flash_rt>;

    shared_flash_runtime get_create_runtime( std::string rt_key );

    bool runtime_exists( std::string );

    std::vector<shared_flash_runtime>
    get_all_runtimes_by( const std::string& );

    shared_flash_runtime get_runtime_by_kname( std::string );

    shared_flash_runtime get_runtime_by_fmem( std::string );

    void register_kernel(std::string, const kernel_desc& );

    void register_fmem( std::string, std::string, const te_variable& );

    bool kernel_exists( std::string, std::string, std::string ); //runtime_key, kname, kimpl


  private:

    void _customize_runtime( std::string );

    bool _transfer_buffer( std::string,shared_flash_runtime, 
                           shared_flash_runtime, te_variable& );
 
    std::string _get_base_loc( std::string, std::string );
      
    shared_flash_runtime _get_runtime_by( std::string );
    
    std::multimap<std::string, resource> _resources;
    std::map<std::string, shared_flash_runtime>   _runtime_ptrs;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

class flash_rt
{

  using FlashableRuntimeInfo = FlashableRuntimeMeta<IFlashableRuntime>;

  public:

    friend runtimes_resource_tracker::shared_flash_runtime 
           runtimes_resource_tracker::get_create_runtime( std::string );

    std::optional<std::string> get_runtime_key() { return _runtime_key; }

    static std::shared_ptr<flash_rt> get_runtime( std::string="" );
 
    status register_kernels( size_t, kernel_t [], std::string [], 
                             std::optional<std::string> [],
                             std::optional<std::string> []  );

    status execute( runtime_vars, uint, std::vector<te_variable>, 
                    std::vector<size_t>, options ); 
   
    status allocate_buffer( te_variable& );

    

    ulong create_transaction();

    status process_transaction( ulong );


  private:
  
    flash_rt( std::string );

    std::shared_ptr<flash_rt> _customize_runtime( std::string );

    void _register_kernel(std::string rtk, const kernel_desc& );

    status _transfer_buffer( std::string, std::shared_ptr<flash_rt>,
                             std::shared_ptr<flash_rt>, te_variable& );

    std::string _recommend_runtime(const std::string &,
                                   const std::vector<te_variable>& );

    void _manage_buffers( std::string, std::string, std::vector<te_variable>& );

    std::optional<std::string>           _runtime_key;
    std::optional<FlashableRuntimeInfo>  _backend;
    std::shared_ptr<IFlashableRuntime>   _runtime_ptr;


    std::multimap<ulong, subaction> _transactions;

    inline static std::shared_ptr<flash_rt> _global_ptr; 
    
    inline static runtimes_resource_tracker _rtrs_tracker;
};


