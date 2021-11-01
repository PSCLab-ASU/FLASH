#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <variant>
#include <utils/common.h>
#include <flash_runtime/transaction_interface.h>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>

#pragma once

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

    std::vector<shared_flash_runtime>
    get_all_runtimes( )
    {
      std::string _;
      return get_all_runtimes_by(_);
    }

    shared_flash_runtime get_runtime_by_kname( std::string );

    shared_flash_runtime get_runtime_by_mem( std::string );

    void register_kernel(std::string, const kernel_desc& );

    void register_mem( std::string, std::string, const te_variable& );

    bool kernel_exists( std::string, const kernel_desc& ); //runtime_key, kname, kimpl
  
    bool transfer_buffers( o_string, o_string, te_variables );

    std::string get_base_loc( std::string, std::string );

  private:

    void _customize_runtime( std::string );

      
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

    static std::shared_ptr<flash_rt> get_runtime( std::string );
 
    status register_kernels( size_t, kernel_t [], std::string [], 
                             std::optional<std::string> [],
                             std::optional<std::string> []  );

    status execute( runtime_vars, uint, std::vector<te_variable>, 
                    std::vector<size_t>, options& ); 
   
    status allocate_buffer( te_variable& );

    auto wait( ulong wid ) 
    {
      return _runtime_ptr->wait( wid );
    }   

    ulong create_transaction();

    status process_transaction( ulong );

    std::shared_ptr<IFlashableRuntime> get_backend() 
    {
      return _runtime_ptr;
    }


  private:
  
    flash_rt( std::string );

    std::shared_ptr<flash_rt> _customize_runtime( std::string );

    void _try_register_kernel(std::vector<kernel_desc>&, 
                              std::optional<std::string> );

    std::string _recommend_runtime(const std::string &,
                                   const std::vector<te_variable>& );

    std::function<int()> 
    _manage_buffers( std::string, std::string, std::vector<te_variable>& );

    std::optional<std::string>           _runtime_key;
    std::optional<FlashableRuntimeInfo>  _backend;
    std::shared_ptr<IFlashableRuntime>   _runtime_ptr;

    transaction_interface _trans_intf;

    inline static std::shared_ptr<flash_rt> _global_ptr; 
    
    inline static runtimes_resource_tracker _rtrs_tracker;
};


