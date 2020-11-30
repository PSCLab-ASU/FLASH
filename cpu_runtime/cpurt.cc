#include "cpurt.h"
#include <iostream>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <climits>
#include <numeric>
#include <chrono>
#include <dlfcn.h>
#include <utility>

/* Registers the factory with flash factory*/
bool cpu_runtime::_registered = FlashableRuntimeFactory::Register(
                                cpu_runtime::get_factory_name(),
                                cpu_runtime::get_runtime() );

std::shared_ptr<cpu_runtime> cpu_runtime::_global_ptr;


size_t get_indices( int ind)
{
  //get current thread id;
  std::thread::id tid = std::this_thread::get_id();

  auto& subact_ctx = cpu_runtime::get_current_job();

  return subact_ctx.get_dim_index(tid, ind );

}


FlashableRuntimeMeta<IFlashableRuntime> cpu_runtime::get_runtime()
{
  //automatic polymorphism to base classa
  FlashableRuntimeMeta<IFlashableRuntime> out{ (std::shared_ptr<IFlashableRuntime> (*)()) get_singleton, 
                                                get_factory_desc() };

  return out;
}

std::shared_ptr<cpu_runtime> cpu_runtime::get_singleton()
{
  std::cout << "entering " << __func__ << std::endl;

  if( _global_ptr ) return _global_ptr;
  else return _global_ptr = std::shared_ptr<cpu_runtime>( new cpu_runtime() );

}

cpu_runtime::cpu_runtime()
{
  std::cout << "entering " << __func__ << std::endl;
  //get number of cores;
  _core_cnt = std::thread::hardware_concurrency();


  //auto test = std::jthread(&cpu_runtime::_thread_main, this, stoken);
  auto thread_obj = std::bind( &cpu_runtime::_thread_main, 
                               this, std::placeholders::_1 );

  for( auto i : std::views::iota((uint)0, _core_cnt) )
  {
    _thread_group.emplace_back( thread_obj );  
  } 

  //ADD ANY LOGIC HERE

  //start the thread group
  _thread_start.notify_all();

}

void cpu_runtime::_thread_main( std::stop_token stop )
{
  std::cout << "entering " << __func__ << std::endl;

  //get current thread id;
  std::thread::id tid = std::this_thread::get_id();

  //hold here until construction is finished
  _lock_wait( [](){ return true; } );
  
    ///forever thrad
  while( !stop.stop_requested() )
  {
    //check if thier any jobs in the subaction table
    _lock_wait( [](){ return cpu_runtime::subaction_exists(); } );
    {
      _stagger_start();
      //i_empty should be padded by the number of parallel processing units
      auto& cur_job = get_current_job();

      while( !cur_job.finished() ) 
      {
        cur_job.exec_work_item();
        cur_job.set_and_decr_index( tid );
        //there are subaction in queue
        //and thier are still work items in the subactions
      }

      _threads_inprog--;
      if( _threads_inprog.load() == 0) _thread_start.notify_all();

      _lock_wait( [&](){ return _threads_inprog.load() != 0; } );
    }

  }

}

status cpu_runtime::wait( ulong  wid)
{
 
  auto& subact = get_subaction_ctx( wid );

  //wait for job to complete
  _lock_wait( [](){ return true; } );

  //remove entry from list
  remove_subaction( wid );

  //reload threads pool
  _threads_inprog.store(_core_cnt);

  //notify thread pool of completion of removal 
  _thread_start.notify_all();

  if( subaction_exists() )  _thread_start.notify_all();
  
  std::cout << "Completed wid : " << wid << std::endl;

  return status{};
}

status cpu_runtime::execute(runtime_vars rt_vars, uint num_of_inputs, 
                            std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms)
{
  std::cout << "entering " << __func__ << std::endl;

  std::string kernel_name = rt_vars.kernel_name_override.value_or(rt_vars.lookup);
  ulong wid = random_number();

  auto func_ptr = _find_function_ptr( kernel_name, rt_vars.kernel_impl_override );

  std::ranges::transform(_thread_group, std::back_inserter(_thread_group_ids),
		         std::identity{}, &std::jthread::get_id );

  auto subact_ctx = subaction_context(  wid, _thread_group_ids, kernel_args, exec_parms );

  subact_ctx.add_kernel_impl( func_ptr );

  bool work_exists = subaction_exists();

  _add_subaction( std::move(subact_ctx) );

  if( !work_exists )  
  {
    _threads_inprog.store(_core_cnt);
    _thread_start.notify_all();
  }

  return status{0, wid};
}

status cpu_runtime::register_kernels( const std::vector<kernel_desc>& kds ) 
{
  
  std::cout << "entering " << __func__ << std::endl;
  
  for(auto kd : kds ) _kernel_repo.check_and_register( kd );

  return {}; 
}

void * cpu_runtime::_find_function_ptr( std::string kernel_name, std::optional<std::string> kernel_impl )
{
  return _kernel_repo.find_kernel_fptr( kernel_name, kernel_impl );
}

void cpu_runtime::_stagger_start()
{
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(1, 100);
  auto delay = std::chrono::milliseconds( distrib(gen) );
  std::this_thread::sleep_for(delay);
}

void exec_repo::check_and_register( kernel_desc& kd )
{
  auto kdef = kd.get_kernel_def();

  if( kdef )
  {
    std::string impl = kdef.value();
    auto exec = std::ranges::find( _programs, impl, &cpu_exec::get_impl );

    if( exec == _programs.end() )
    {
      void * main_h = dlopen( kdef.value().c_str(), RTLD_LAZY );
      if( main_h )
      {
	std::cout << "Added " << impl << " to cpu repo" << std::endl;
        _programs.emplace_back(impl, main_h);
      } 
      else 
      {
        std::cout << "Could not register binary " << impl << std::endl;
	return;
      }

      exec = std::ranges::find( _programs, impl, &cpu_exec::get_impl);
    }

    if( exec->check_kernel( kd.get_kernel_name() ) ) 
    {
      
      auto mname = exec->get_mangled_name( kd.get_kernel_name() ); 
      if( mname )
      {
        void * func_ptr = dlsym( exec->get_bin_ptr(), mname->c_str() );

        if( func_ptr != NULL ) 
        {
          std::cout << "adding " << kd.get_kernel_name()  
                    << "(" << kd.get_kernel_name() << ") to cpu repo" << std::endl;

          exec->insert_kernel( kd.get_kernel_name(), mname, func_ptr);

        }

      } else std::cout <<"Could not find kernel function " <<  kd.get_kernel_name() 
	  	       <<" in " << impl <<  std::endl;
    }  //end of kernel creation
  } //binary esists
}

void * exec_repo::find_kernel_fptr( std::string kernel_name , std::optional<std::string> kernel_impl )
{
  //TBD
   auto bin_filt = [&](auto exec)
   {
     if( kernel_impl) 
       return (exec.get_impl() == kernel_impl.value() ) && 
              ( exec.check_kernel(kernel_name) );
     else
       return exec.check_kernel(kernel_name);
   }; 
  //
  auto bin_kernel = [&](auto exec)
  {
    return exec.find_kernel( kernel_name );
  };
  
  auto kernels = _programs | std::views::filter( bin_filt ) | std::views::transform( bin_kernel );

  for( auto kernel : kernels)
  {
    return kernel.get_func_ptr();
  }
  
  return nullptr;	   
} 

cpu_exec::cpu_exec( std::string impl, void * bin_ptr)
{
  _impl = impl; 
  _bin_ptr = bin_ptr;  
  _header = _get_symtable_hdr();
}


std::string cpu_exec::get_impl()
{
  return _impl;
}

void * cpu_exec::get_bin_ptr() 
{ 
  return _bin_ptr; 
}

cpu_kernel& cpu_exec::find_kernel( std::string kernel_name )
{
  auto iter1 = std::ranges::find( _kernels, kernel_name, &cpu_kernel::get_kname );
  auto iter2 = std::ranges::find( _kernels, kernel_name, &cpu_kernel::get_mkname);

  if( iter1 != _kernels.end() )     return *iter1;
  else if(iter2 != _kernels.end() ) return *iter2;
  else
  {
    std::cout << "Could not reconcile kernels" << std::endl;
    throw std::runtime_error("Could not find kernel");
  }

}

bool cpu_exec::check_kernel( std::string kernel_name)
{
  auto iter1 = std::ranges::find( _kernels, kernel_name, &cpu_kernel::get_kname );
  auto iter2 = std::ranges::find( _kernels, kernel_name, &cpu_kernel::get_mkname);

  return ( iter1 != _kernels.end() ) || ( iter2 != _kernels.end() );
  
}

void cpu_exec::insert_kernel( std::string kname, std::optional<std::string> mkernel_name, void * func_ptr)
{ 
  std::string kernel_name = mkernel_name.value_or(kname);
  std::string mkern = mkernel_name.value_or("");

  auto k = check_kernel( kernel_name );
  if( !k )
  {
    _kernels.emplace_back( kernel_name, mkern, func_ptr);
  }
  else std::cout << "Kernel already exists" << std::endl;
}

Elf64_Shdr cpu_exec::_get_symtable_hdr()
{
  FILE *file = fopen( get_impl().c_str(), "rb" );
  Elf64_Ehdr elf_header;
  Elf64_Shdr header;

  auto elf_func = [&](auto func, auto ... parms)->bool 
  {
    bool b = func( parms ...);
    fseek(file, 0, SEEK_SET);
    return b;
  };

  if( file != nullptr )
  {
    bool isElf = elf_func( elf_is_elf64, file);
    if( isElf )
    {
      bool symtab_exists = elf_func(elf64_get_symtab_section, 
                                    file,
                                    (const Elf64_Ehdr *) &elf_header, 
                                    &header);       
   
      if( symtab_exists )
      {
        std::cout << "Found symtab..." << std::endl;

      } else std::cout << "Could not find symbol table " << std::endl;

    } else std::cout << "Is not valid ELF file " << std::endl;

  } else std::cout << "File doesn't exists" << std::endl;
  //closing file
  fclose(file);

  return header;
}

std::optional<std::string> cpu_exec::get_mangled_name( std::string kernel_name )
{
  auto cdata = std::string( &((char *) _bin_ptr)[_header.sh_offset], _header.sh_size); 

  std::smatch sof_match;
  std::string func_regex;
  std::string delimiter = "([^\\. ]+)?";

  //foudn exact name match
  if( _test( kernel_name ) ) return kernel_name;

  //parse namepsaces and construct function regex
  std::regex_search(kernel_name, sof_match, std::regex("[^::]+") );
  func_regex = std::accumulate( std::begin(sof_match), std::end(sof_match), 
                                delimiter, [&](auto prev, auto cur)
                                {
                                  return (prev + cur.str() + delimiter );
                                } );

  std::cout << "Printing " << func_regex << std::endl;

  if( std::regex_search(cdata, sof_match, std::regex(func_regex) ) )
  {

    for( auto match : sof_match )
    {
      std::cout << "Checking function : " << match << std::endl;
      if( _test(match) )
      {
        std::cout << "Successfuly found entrypoint for ..." << match <<  std::endl;
        return match;
      } //end of loading function
      else std::cout << "No match found" << std::endl;
    }
  }

  return {};
}

bool cpu_exec::_test( std::string func_name )
{
  void * func_ptr = dlsym( get_bin_ptr(), func_name.c_str() );

  return (func_ptr != NULL);

}

subaction_context::subaction_context( ulong wid, std::vector<std::thread::id>& threads, std::vector<te_variable> k_args, std::vector<size_t> max_cnts )
: _wid(wid), _kernel_args(k_args), _next_index(threads.size()), _padding(threads.size())
{
  
  using table_t = std::vector< std::vector<size_t > >;

  auto len = std::accumulate( max_cnts.begin(), max_cnts.end(), 1, std::multiplies<size_t>{} ) + 1;

  _index_table = table_t(len, std::vector<size_t>(max_cnts.size(), -1) );
		        
  //generate index table 
  std::partial_sum( _index_table.begin(), _index_table.end(), _index_table.begin(), 
                   [&](const std::vector<size_t>& past, const std::vector<size_t>& current)
	           {     
	             std::vector<size_t> out = past;               
		     for(size_t i=0; i < max_cnts.size(); i++)
		     {
		       if( past[i] < (max_cnts[i]-1) )
		       {
		         out.at(i) += 1;
			 break;
		       }
		       else
		       {
		         out.at(i) =0;   
	               }
                     }
                     std::ranges::copy( out, std::ostream_iterator<size_t>(std::cout, ", ") );
		     std::cout << std::endl;
		     return out;
                  } ); 
 
  //adding panding
  auto table_padding = table_t(_padding, std::vector<size_t>(max_cnts.size(), 0) );
  //add padding
  _index_table.insert( _index_table.begin(), table_padding.begin(), table_padding.end() );
  
  //prefill
  for(uint i=0; i < _padding; i++) _ci_per_thread.emplace(threads[i], i);
 

}

subaction_context& subaction_context::operator=( const subaction_context& )
{
  //TBD
  std::cout << "NEED TO COMPLETE MOVE CTOR subaction_context" << std::endl;
  return *this;
}

subaction_context::subaction_context( const subaction_context&& )
{
  //TBD
  std::cout << "NEED TO COMPLETE MOVE CTOR subaction_context" << std::endl;
}

ulong subaction_context::get_wid()
{
  return _wid;
}

void subaction_context::add_kernel_impl( void * func_ptr)
{
  _func_ptr = func_ptr;
}

bool subaction_context::finished()
{
  return _finished;
}

//maybe torn reads and writes?
void subaction_context::set_and_decr_index(std::thread::id tid)
{
  _ci_per_thread.at(tid) = _next_index--;
 
  //if any thread gets within the padded region everything is complete;
  _finished = _next_index < _padding;
}
 
size_t subaction_context::get_dim_index( std::thread::id& tid, int& ind) 
{
  auto table_index = _ci_per_thread.at(tid); 
  return _index_table[table_index][ind];
}

void subaction_context::exec_work_item()
{
  //I hope the branch predictor is optimized 
  if( _kernel_args.size() == 0 )
    execute_ninput_method<0>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 1 )
    execute_ninput_method<1>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 2 )
    execute_ninput_method<2>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 3 )
    execute_ninput_method<3>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 4 )
    execute_ninput_method<4>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 5 )
    execute_ninput_method<5>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 6 )
    execute_ninput_method<6>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 7 )
    execute_ninput_method<7>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 8 )
    execute_ninput_method<8>( _func_ptr, _kernel_args); 
  else if( _kernel_args.size() == 9 )
    execute_ninput_method<9>( _func_ptr, _kernel_args); 
  else
    std::cout << "To many argument not supported" << std::endl;
    
}
