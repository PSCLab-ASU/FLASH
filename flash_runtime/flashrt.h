#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
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


class flash_rt
{

  using FlashableRuntimeInfo = FlashableRuntimeMeta<IFlashableRuntime>;

  public:

    static std::shared_ptr<flash_rt> get_runtime( std::string="" );
 
    status register_kernels( size_t, kernel_t [], std::string [], std::optional<std::string> [] );

    status execute( runtime_vars, uint, std::vector<te_variable>, std::vector<size_t>, options ); 
   
    ulong create_transaction();

    status process_transaction( ulong );


  private:
  

    flash_rt( std::string );

    std::optional<FlashableRuntimeInfo>  _backend;
    std::shared_ptr<IFlashableRuntime>   _runtime_ptr;


    std::multimap<ulong, subaction> _transactions;

    static std::shared_ptr<flash_rt> _global_ptr; 
    
};



