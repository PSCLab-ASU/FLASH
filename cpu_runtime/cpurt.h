#include <memory>
#include <iostream>
#include <common.h>
#include <vector>
#include <map>
#include <thread>
#include <utility>
#include <mutex>
#include <regex>
#include <condition_variable>
#include <flash_runtime/flash_interface.h>
#include <flash_runtime/flashable_factory.h>
#include <boost/align/aligned_allocator.hpp>

#include "elf.c"

#pragma once

size_t get_indices( int );

//template the _next_index with atomic to see how the performance fairs

struct subaction_context
{
  using index_t = std::vector<size_t>;

  subaction_context( ulong, std::vector<std::thread::id>&, 
		     std::vector<te_variable>, std::vector<size_t> );

  subaction_context( const subaction_context&& );

  subaction_context& operator=( const subaction_context& );

  ulong get_wid();

  void add_kernel_impl( void * func_ptr);
   
  bool finished();

  //maybe torn reads and writes?
  void set_and_decr_index(std::thread::id tid);
 
  size_t get_dim_index( std::thread::id& tid, int& ind);

  void exec_work_item();

  ////////////////////////////members///////////////////////////////
  ulong _wid;
  void * _func_ptr;
  std::vector<te_variable> _kernel_args;
  std::map<std::thread::id, size_t> _ci_per_thread;
  std::vector< index_t > _index_table;
  size_t _next_index;
  bool _finished =false;
  unsigned int _padding;

};

struct cpu_kernel
{
  std::string _kernel_name;
  std::string _mangled_name;

  void * _func_ptr;

  cpu_kernel( std::string kname, std::string mkname, void * func_ptr)
  {
    _kernel_name  = kname;
    _mangled_name = mkname;
    _func_ptr     = func_ptr;  
  }

  cpu_kernel( const cpu_kernel& rhs )
  {
    _kernel_name  = rhs.get_kname();
    _mangled_name = rhs.get_mkname();
    _func_ptr     = rhs.get_func_ptr();  
    
  }

  cpu_kernel( const cpu_kernel&& rhs )
  {
    _kernel_name  = rhs.get_kname();
    _mangled_name = rhs.get_mkname();
    _func_ptr     = rhs.get_func_ptr();  
    
  }

  std::string get_kname() const
  {
    return _kernel_name;
  };

  std::string get_mkname() const
  {
    return _mangled_name;
  };

  void * get_func_ptr() const
  {
    return _func_ptr;
  }

};

struct cpu_exec
{
  std::string _impl;
  void * _bin_ptr;  //this is the mmap of the file
  Elf64_Shdr _header;

  std::vector<cpu_kernel> _kernels;

  cpu_exec( std::string impl, void * bin_ptr);

  std::string get_impl();

  void * get_bin_ptr();

  cpu_kernel& find_kernel( std::string kernel_name );

  bool check_kernel( std::string );

  void insert_kernel( std::string, std::optional<std::string>, void * );

  std::optional<std::string> get_mangled_name( std::string );

  Elf64_Shdr _get_symtable_hdr();

  bool _test( std::string );
};

struct exec_repo
{
  std::vector<cpu_exec> _programs;

  void check_and_register( kernel_desc& );

  void * find_kernel_fptr( std::string, std::optional<std::string> );

};

class cpu_runtime : public IFlashableRuntime
{

  public:

    status register_kernels( const std::vector<kernel_desc> & ) final;

    status execute( runtime_vars, uint, std::vector<te_variable>, std::vector<size_t> ) final;  

    status wait( ulong ) final;

    static FlashableRuntimeMeta<IFlashableRuntime> get_runtime();

    static std::shared_ptr<cpu_runtime> get_singleton();

    static std::string get_factory_name() { return "ALL_CPU"; }

    static std::string get_factory_desc() { return "This runtime support CPU-based accelerators"; }

    //subaction ID with context
    static std::vector<subaction_context > g_subaction_table;

    static bool subaction_exists()
    {
      return g_subaction_table.size() > 0;
    }
    
    static subaction_context& get_current_job()
    {
      return *g_subaction_table.begin();
    }

    static void pop_current_job()
    {
      g_subaction_table.erase( g_subaction_table.begin() );
    }

    static subaction_context& get_subaction_ctx( ulong wid )
    {
      auto ctx = std::ranges::find(g_subaction_table, wid, &subaction_context::get_wid );
      return *ctx;
    }

    static void remove_subaction( ulong wid )
    {
      auto rem_e = std::ranges::find(g_subaction_table, wid, &subaction_context::get_wid );
 
      g_subaction_table.erase(rem_e);

    }

  private:

    cpu_runtime();
 
    void _thread_main( std::stop_token );

    void _lock_wait( auto pred)
    {
      std::unique_lock<std::mutex> lk(_mu);
      _thread_start.wait( lk, pred );
      lk.unlock();
    }
 
    void _stagger_start();

    void _add_subaction( subaction_context&& ctx )
    {
      g_subaction_table.push_back(std::forward<subaction_context>( ctx ) );
    }

    void * _find_function_ptr( std::string, std::optional<std::string> );

    static  std::shared_ptr<cpu_runtime> _global_ptr; 

    static bool _registered;

    unsigned int _core_cnt;

    std::atomic_int                _threads_inprog;
    std::vector< std::jthread >    _thread_group;
    std::vector< std::thread::id > _thread_group_ids; //could I use a view?

    exec_repo _kernel_repo;

    std::condition_variable _thread_start;
    std::mutex _mu;
    
};



