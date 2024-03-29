#include "cuda_runtime/flash_cuda.h"
#include <iostream>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <climits>
#include <sys/mman.h>
#include <sys/stat.h> 
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <regex>
#include <vector>
#include <chrono>

using namespace std::chrono_literals;

/* Registers the factory with flash factory*/
bool flash_cuda::_registered = FlashableRuntimeFactory::Register(
                                flash_cuda::get_factory_name(),
                                flash_cuda::get_runtime() );

std::shared_ptr<flash_cuda> flash_cuda::_global_ptr;


FlashableRuntimeMeta<IFlashableRuntime> flash_cuda::get_runtime()
{
  //automatic polymorphism to base classa
  FlashableRuntimeMeta<IFlashableRuntime> out{ (std::shared_ptr<IFlashableRuntime> (*)()) get_singleton, 
                                                get_factory_desc() };

  return out;
}


std::shared_ptr<flash_cuda> flash_cuda::get_singleton()
{

  if( _global_ptr ) return _global_ptr;
  else return _global_ptr = std::shared_ptr<flash_cuda>( new flash_cuda() );

}

flash_cuda::flash_cuda()
: _kernel_lib{ _kernel_comps }
{
  std::cout << "Ctor'ing flash_cuda...." << std::endl;
  //initialize driver API
  cuInit(0);
  //grab a handle to each devices primary context
  _retain_cuda_contexts();


}

status flash_cuda::allocate_buffer( te_variable& arg, bool& success )
{

  auto dir         = arg.parm_attr.dir;
  auto is_flash    = arg.parm_attr.is_flash_mem;
  auto host_buffer = arg.get_data();
  auto sz          = arg.get_bytes();
  auto md = mem_detail( host_buffer, sz, is_flash );

  std::string key = arg.get_mem_id();

  _mem_registry.emplace(key, md );

  success = true;

  return status{};
}

//only call after transaction is complete
status flash_cuda::deallocate_buffer( std::string buffer_id, bool& success)
{
  auto md = _mem_registry.at(buffer_id);

  //deallocate 
  md.deallocate_all( true );

  _mem_registry.erase(buffer_id);

  success = true; 

  return status{};
}

status flash_cuda::deallocate_buffers( std::string tid )
{

  for(size_t i =_mem_registry.size(); i > 0; i--)
  {
    auto iter = std::next( _mem_registry.begin(), i);
    auto[key, md] = *iter;

    if( !md.is_flash() && (md.last_tid() == tid) )
      _mem_registry.erase(iter); 
  }

  return status{};
}

status flash_cuda::transfer_buffer( std::string buffer_id, void * host_buffer)
{

  auto md = _mem_registry.at(buffer_id);
  
  md.transfer_lastDtoH();

  return status{};
}

//status flash_cuda::set_trans_intf( std::shared_ptr<transaction_interface> trans_intf )
status flash_cuda::set_trans_intf( transaction_interface& trans_intf )
{
  _trans_intf = &trans_intf;
  return status{};
}

status flash_cuda::register_kernels( const std::vector<kernel_desc>& kds, 
                                     std::vector<bool> & successes ) 
{
  
  std::cout << "calling flash_cuda::" << __func__<<  std::endl;
  bool found=false;
  for( auto kernel : kds )
  {
    
    if( kernel._kernel_type == kernel_t::INT_BIN ||
        kernel._kernel_type == kernel_t::EXT_BIN )
    {
      if( kernel._kernel_definition ) 
        printf("->registering %s in %s \n", kernel._kernel_name.value().c_str() 
	      , kernel._kernel_definition.value().c_str() );
      else
        printf("registering : %s \n", kernel._kernel_name.value().c_str() );

      printf("_process_external_binary...\n");
      _process_external_binary( kernel, found );
      successes.push_back(found);
    }
    else
      std::cout << "registration format not supported" << std::endl;
  }

  //reset back to initial context
  _reset_cuda_ctx();

  return {}; 
}

bool flash_cuda::_try_exec_next_pjob( ulong wid )
{
  bool ret = true;
  auto& pjob = _pending_jobs.at(wid);
  auto[trans_id, subaction_id] = pjob.get_ids();

  //////////////////////////////////////////////////////////////////////////////
  auto& sa_payload = _trans_intf->find_sa_within_ta(subaction_id, trans_id);
  sa_payload.post_pred();
  //////////////////////////////////////////////////////////////////////////////
   //TBD
   //need to check for the second job with the given tid
   //because the first one hasn't been deleted yet.a
   //if the second TID entry doesn't have a deffered action
   //return false
  auto pjob_pipe = _pending_jobs | std::views::filter([&](auto& pj) 
  {
    auto[tid, sa_id] = pj.second.get_ids();
    return tid == trans_id;
  } ) | std::views::drop(1) | std::views::take(1);

     
  if( auto nx_pjob_It = std::begin(pjob_pipe); 
      (nx_pjob_It != std::end(pjob_pipe)) )
  {
    auto[nx_tid, nx_sid] = (*nx_pjob_It).second.get_ids(); 
    auto& nx_sa_payload = _trans_intf->find_sa_within_ta(nx_sid, nx_tid);

    nx_sa_payload.pre_pred();

    if( (*nx_pjob_It).second.pending_launch )
      (*nx_pjob_It).second.pending_launch();

    ret = false;
  }

  return ret;
}


void flash_cuda::_conclude_buffer_placement( pending_job_t& pjob)
{

  auto tid   = pjob.trans_id;
  auto sa_id = pjob.subaction_id; 
	
  //check if transaction defers deallocation
  // or if they are flash memory 
  bool defer_dealloc   = _trans_intf->operator()(tid).check_option(sa_id, trans_options::DEFER_DEALLOC  );
  bool defer_writeback = _trans_intf->operator()(tid).check_option(sa_id, trans_options::DEFER_WB );
  bool g_defer_out_dealloc = _trans_intf->operator()(tid).check_option(sa_id, global_options::DEFER_OUTPUT_DEALLOC );

  printf("defer_dealloc = %d\n",   defer_dealloc);
  printf("defer_writeback = %d\n", defer_writeback);
  printf("g_defer_dealloc = %d\n", g_defer_out_dealloc);
   
  for( auto& tvar : pjob.kernel_args )
    if( tvar.is_flash_mem() )
    {
      //if flash memor skip implicit processing
    }
    else if( !defer_writeback && defer_dealloc )
    {
      //forces a writeback at the end of the transaction
      //bust dont deallocate buffer
      _writeback_to_host( tid, tvar );
    }
    else if( defer_writeback && !defer_dealloc )
    {
      //just edallocate buffer without writing the data back
      _deallocate_buffer( tid, DIRECTION::IN, tvar );
      _deallocate_buffer( tid, DIRECTION::INOUT, tvar );

      if( !g_defer_out_dealloc )
      {
        _writeback_to_host( tid, tvar );
        _deallocate_buffer( tid, DIRECTION::OUT, tvar );
      }
    }
    else if( !defer_writeback && !defer_dealloc )
    {
      //write back to host and deallocate buffer
      //just edallocate buffer without writing the data back
      _writeback_to_host( tid, tvar );
      _deallocate_buffer( tid, DIRECTION::IN, tvar );
      _deallocate_buffer( tid, DIRECTION::INOUT, tvar );

      if( !g_defer_out_dealloc )
        _deallocate_buffer( tid, DIRECTION::OUT, tvar );
    }

    
}

void flash_cuda::_notify_all_pjobs()
{


}

status flash_cuda::wait( ulong  wid)
{
  std::cout << "Enttering " << __func__ << std::endl;
  //get job from pending jobs  
  auto entry = _pending_jobs.find( wid);
  printf("----- Mark 1-------\n");

  if( entry != _pending_jobs.end() ) 
  {
    printf("----- Mark 2-------\n");
    //set context to the job related to wid;
    auto pjob  = entry->second;
    auto tid   = pjob.trans_id;
    auto sa_id = pjob.subaction_id; 
    printf("----- Mark 3-------\n");
    //////////////////////////////////////////////////////////////////////////////
    auto& sa_payload = _trans_intf->find_sa_within_ta(sa_id, tid);
    auto [num_of_inputs, rt_vars, kernel_args, 
          exec_parms, pre_pred, post_pred] = sa_payload.input_vars();
    auto is_last_sa = sa_payload.is_last();
    printf("----- Mark 4-------\n");
    //////////////////////////////////////////////////////////////////////////////
    //This function 
    //1} check to see if there is subactions before the current
    //subaction that also has overlapping write buffers
    //if there is "wait" else, carry on
    pre_pred();
    printf("----- Mark 5-------\n");

    _kernel_comps.set_active_context( pjob.cuCtx_key );
    //blocks until job is complete
    std::cout << "Sync'ing ctx..." << std::endl;
    printf("----- Mark 6-------\n");
    if (sa_id == 0 )
      int ret = cuCtxSynchronize();
    else{
      if( pjob.pending_launch )
      {
        printf("----- Mark 6.5-------\n");
        pjob.pending_launch();
        int ret = cuCtxSynchronize();
      } 
      else 
      {
        printf("No pending jobs...");
	return status{-1};
      }
    }
    printf("----- Mark 7-------\n");

    std::cout << "waiting complete" << std::endl;
    ///////////////release resources/////////////////
    post_pred();
    std::cout << "Releasing device buffers" << std::endl;
    _release_device_buffers( tid, kernel_args );
    /////////////////////////////////////////////////
    std::cout << "Released device buffers" << std::endl;
    sa_payload.mark_complete();

    _notify_all_pjobs();
    //This means there are no more pending jobs
    //or a specific transaction
    std::cout << "Processing last subaction..." << std::endl;
    _conclude_buffer_placement( pjob );

    //remove job
    _pending_jobs.erase( wid);
  }
  else std::cout << "Could not find pending job..." << std::endl;
 
  //return status{};
  return status{1,1};
}

//status flash_cuda::execute(runtime_vars rt_vars, uint num_of_inputs, 
//                              std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms)
status flash_cuda::execute( ulong trans_id, ulong sa_id )
{
  printf("\nExecuting form cuda-flash_runtime... : tid = %llu, sid = %llu \n", trans_id, sa_id );
  //////////////////////////////////////////////////////////////////////////////
  ulong wid = random_number(); 
  std::vector<CUdeviceptr> device_buffers;
  std::function<std::vector<ull>()> deferred_launch;

  auto& sa_payload = _trans_intf->find_sa_within_ta(sa_id, trans_id);
  auto [num_of_inputs, rt_vars, kernel_args, 
         exec_parms, pre_pred, post_pred] = sa_payload.input_vars();

  auto sa_kattrs = sa_payload.get_kattrs();
  bool need_table = sa_payload.need_index_table();

  //////////////////////////////////////////////////////////////////////////////
  auto kname = rt_vars.get_kname_ovr().value_or( rt_vars.get_lookup() );
  //set current context to context with the least amount of work
  _set_least_active_gpu_ctx();
  std::string kernel_name = rt_vars.kernel_name_override?rt_vars.kernel_name_override.value():rt_vars.get_lookup();
  std::optional<CUfunction> func_binary;
  if( rt_vars.kernel_impl_override )
  {
    func_binary = _kernel_lib.get_cufunction_for_current_ctx( kernel_name, rt_vars.kernel_impl_override );
  }
  else
  {
    func_binary = _kernel_lib.get_cufunction_for_current_ctx( kernel_name, {} );
  }

  auto ctx_key     = _kernel_comps.get_current_ctx_key();
  int ret = CUDA_SUCCESS;
  //auto input_kargs = std::vector<te_variable>( kernel_args.begin(), kernel_args.begin() + num_of_inputs);
    //launch kernel
  std::vector<size_t> dims;
  std::ranges::copy(exec_parms, std::back_inserter(dims) );

  o_string table_id;
  if( need_table ) table_id = index_table::preview_table_id( dims );

  if( func_binary )
  {
    printf("<--Found binary--> : %i\n", sa_payload.is_first() );
    int i =0;
    bool defer=false;

    std::ranges::copy( dims, std::ostream_iterator<size_t>{std::cout, ", "} );
    std::cout << std::endl;


    if( sa_payload.is_first() ) //eager
    {
      printf("Found the first subaction\n");

      pre_pred();   
      printf("completed pre_pred subaction\n");
      _assess_mem_buffers( trans_id, sa_id, kernel_args, dims, sa_kattrs );
      printf("Completed memstate construction....\n");

      //std::vector<CUdeviceptr> device_buffers = _checkout_device_buffers( trans_id, kernel_args, table_id, defer );
      device_buffers = _checkout_device_buffers( trans_id, kernel_args, table_id, defer );
      printf("Number of device buffers checkedout = %i \n", device_buffers.size() );
      deferred_launch = [device_buffers](){
        return device_buffers;
      }; 
      //create pending job entry
      //WARNING MAKE SURE this void * is valid during the invocation of the kernel
      //void * is a CUDA back to erase the type of the device pointer.
      std::vector<void *> void_devs_buffs;
      std::ranges::transform(device_buffers, std::back_inserter( void_devs_buffs ),
      		             [](auto& device_buffer) { return (void *) &device_buffer; } );

      std::vector<CUdeviceptr> indTableIt;
      //key is the offset address of the parititon
      std::map<CUdeviceptr, std::vector<size_t> > dimps;

      if( need_table )
      {
	printf("Execution requires an Index Table with N segments\n");
        auto& table_md = _mem_registry.at( table_id.value() );
	auto table_o = table_md.get_table();
	if( table_o )
	{
	  auto dims = table_o.value().get_dims();
	  printf("Index Table with DIM (%llu, %llu) \n", dims[0], dims[1] );
	}
	auto cit = cuda_index_table( device_buffers.back(), table_md );
        indTableIt = cit.range();
	dimps = cit.get_dim_parts();
	//creating space for 
        //can be a table buffer or a last parm
        //void_devs_buffs.push_back(nullptr);
  
      } 
      else 
      {
	printf("Execution does not requires an Index Table with N\n");
	std::vector<size_t> dim_partition;
	//shortcut to bypass theh loop
	//this makes  the last buffer an argument
        indTableIt.emplace_back( device_buffers.back() );

	std::ranges::copy(dims, std::back_inserter(dim_partition) );
        dimps.emplace(device_buffers.back(), dim_partition ); 	
      }

       
      printf("number of table segments = %llu \n", indTableIt.size() );
      for(int tab_idx=0; auto table_seg : indTableIt )
      {
        // updating base table pointer
        tab_idx++;

	if(need_table)
	  void_devs_buffs.push_back( (void *) &table_seg );

        if( dimps.count(table_seg) )
	{
	  
	  printf("Found table segment := %llu\n", *( (CUdeviceptr *) void_devs_buffs.back()) );
	  auto dimp = dimps.at(table_seg);
          std::fill_n( std::back_inserter(dimp), 7-dimp.size(), 1 );
	  printf("{ %llu,%llu,%llu,%llu,%llu,%llu,%llu } : %llu\n", 
	             dimp[0],  dimp[1], dimp[2], dimp[3], dimp[4], dimp[5], dimp[6], dimp.size() );

          ret |= cuLaunchKernel(func_binary.value(), dimp[0], dimp[1], dimp[2], dimp[3], 
                                dimp[4], dimp[5], dimp[6], nullptr, void_devs_buffs.data(), 0);
          if( ret == CUDA_SUCCESS)
          {
            std::cout << "Kernel successfully launched " << tab_idx << "..." << std::endl;
            _job_count_per_ctx.at(ctx_key) += 1;
          }
          else 
	  {
	    std::cout << "Could not launch " << rt_vars.get_lookup() << " : " << ret << std::endl;
            throw std::runtime_error("Could not launch...");
	  }
          
	}
	else
	  if( tab_idx < 10 )
	    printf("Could not find table segment %llu\n", table_seg);
      }
    }
    else //lazy
    {
      deferred_launch = [&, tid=trans_id, sid=sa_id, 
		            ckey=ctx_key, fbinary=func_binary, 
		            edims=dims, kattrs=sa_kattrs,
			    fname=rt_vars.get_lookup(),
                            tab_id=table_id, exdims=dims]()
      {
        bool defer=false;
        pre_pred();   
        _assess_mem_buffers( tid, sid, kernel_args, exdims, kattrs );
        std::vector<CUdeviceptr> _device_buffers = _checkout_device_buffers( tid, kernel_args, tab_id, defer );

        if(defer) std::logic_error("Buffers are not available");

	printf("\n!!!Launching deferred kernel logic!!!\n");

        //create pending job entry
        //WARNING MAKE SURE this void * is valid during the invocation of the kernel
        //void * is a CUDA back to erase the type of the device pointer.
        std::vector<void *> void_devs_buffs;
        std::ranges::transform(_device_buffers, std::back_inserter( void_devs_buffs ),
      	  	               [](auto& device_buffer) { return (void *) &device_buffer; } );
	printf("\n!!!Transformed deferred device_buffers!!!\n");
        _kernel_comps.set_active_context( ckey );
	printf("\n!!!set deferred active context!!!\n");
      
        std::vector<CUdeviceptr> indTableIt;
        //key is the offset address of the parititon
        std::map<CUdeviceptr, std::vector<size_t> > dimps;

        if( need_table )
        {
	  printf("\n!!!Needed table!!!\n");
          auto& table_md = _mem_registry.at( tab_id.value() );
	  printf("\n!!!Found table!!!\n");
	  auto cit = cuda_index_table( _device_buffers.back(), table_md );
	  printf("\n!!!converted table!!!\n");
          indTableIt = cit.range();
	  printf("\n!!!got range!!!\n");
	  dimps = cit.get_dim_parts();
	  printf("\n!!!got dim parts!!!\n");
	  //creating space for 
          //can be a table buffer or a last parm
          //void_devs_buffs.push_back(nullptr);
  
        } 
        else 
        {
	  std::vector<size_t> dim_partition;
	  //shortcut to bypass theh loop
	  //this makes  the last buffer an argument
          indTableIt.emplace_back( _device_buffers.back() );

	  std::ranges::copy(exdims, std::back_inserter(dim_partition) );
          dimps.emplace(_device_buffers.back(), dim_partition );
        }

        int i=0; 
        for(auto table_seg : indTableIt )
        {
          i++;
	  printf("\n!!!got table parts %i!!!\n", indTableIt.size() );
            //updating base table pointer
  	  void_devs_buffs.back() = (void *) &table_seg;
	  printf("\n!!!got dim segment!!!\n");
          auto dimp = dimps.at(table_seg);
	  printf("\n!!!deferred dimp parts!!!\n");
          std::fill_n( std::back_inserter(dimp), 7-dimp.size(), 0 );
	  printf("\n!!!deferred fill_n!!!\n");

          int res = cuLaunchKernel(fbinary.value(), dimp[0], dimp[1], dimp[2], dimp[3], 
                                   dimp[4], dimp[5], dimp[6], nullptr, void_devs_buffs.data(), 0);
	  printf("\n!!!Deferred Kernel launched!!!\n");
          if( ret == CUDA_SUCCESS)
          {
            std::cout << "Kernel successfully launched "<< i <<"..." << std::endl;
            _job_count_per_ctx.at(ckey) += 1;
          }
          else 
	  {
	    std::cout << "Could not launch " << fname << " : " << ret << std::endl;
            throw std::runtime_error("Could not launch...");
	  }
	}
        return _device_buffers;
      };

    }

    printf("**************\nPending Jobs WID : %i\n*****************\n", wid);
    _pending_jobs.emplace(std::piecewise_construct,
      	                  std::forward_as_tuple(wid),
    	  	          std::forward_as_tuple(trans_id, sa_id, ctx_key, kernel_args,
                          num_of_inputs, deferred_launch) );
    auto stat = status{ret, wid };
    return stat;

  } //end  func_binary
  else
  {
    std::cout << "Could not find function to execute" << std::endl;
  }

  std::cout << "  Last Mark" << std::endl;
  return status{-1};
}

std::vector<CUdeviceptr>
flash_cuda::_checkout_device_buffers( ulong tid, std::vector<te_variable>& kargs, o_string table_id, bool & defer )
{

  printf("      checkout_device_buffers(tid = %llu)\n", tid );

  std::vector<CUdeviceptr> out;

  std::string tid_str = std::to_string(tid);

  std::ranges::for_each(kargs, [&](auto arg)
  {
    std::string key = arg.get_mem_id();
    printf("        checkout_device_buffers:: arg = %s\n", key.c_str() );

    //////////////////////////////////////////////////////
    auto& md = _mem_registry.at(key);
    printf("          Mark 1\n");
    //////////////////////////////////////////////////////
    auto& ms = md.get_mstate( tid_str );
    printf("          Mark 2\n");
    defer |= ms.in_use && (arg.parm_attr.dir != DIRECTION::IN);
    printf("          Mark 3\n");
    ms.in_use = true;

    
    printf("============ ms.dptr.value() = %p\n", ms.dptr.value() );
    out.push_back( ms.dptr.value() );
    printf("          Mark 4\n");
    //////////////////////////////////////////////////////
  } );

  //adding inde table
  if( table_id )
  {
    printf("          Adding table %s\n", table_id.value().c_str() );
    if( _mem_registry.count( table_id.value() ) ) 
    { 
      printf("           Getting memory_description\n"  );
      auto& md = _mem_registry.at( table_id.value() );
      if( md.memstate_exists( tid_str ) )
      { 
        auto& ms = md.get_mstate( tid_str );
        printf("           Getting memory state\n" );
        ms.in_use = true;
        out.push_back( ms.dptr.value() );
      }
      else 
        printf("         Could NOT find a valid memory state\n");
  
    }
    else 
      printf("           Could NOT find memory description\n");
  }

  return out;

}

void
flash_cuda::_release_device_buffers( ulong tid, std::vector<te_variable>& kargs )
{

  std::string tid_str = std::to_string(tid);

  std::ranges::for_each(kargs, [&](auto arg)
  {
    std::string key = arg.get_mem_id();
    //////////////////////////////////////////////////////
    auto& md = _mem_registry.at(key);
    //////////////////////////////////////////////////////
    auto& ms = md.get_mstate( tid_str );
    ms.in_use = false;
    //////////////////////////////////////////////////////
  } );

}

void flash_cuda::_reset_cuda_ctx()
{
  _kernel_comps.contexts[0].set_context_to_current();
}

int flash_cuda::_find_cubin_offset(void * nvFCubinBase, size_t fbin_len, std::string func_name, size_t * cubin_offset, std::string * mangled_name)
{
  std::smatch sof_match;
  CUmodule cuModule;
  CUfunction khw;    
  size_t offset =0;
  auto cdata = std::string( (char *) nvFCubinBase, fbin_len);
  std::vector<size_t> mod_ind;

  std::regex sof_regex ("\x50\xed\x55\xba\x01\x00\x10\x00");
  while(std::regex_search(cdata, sof_match, sof_regex)) 
  {
    for (size_t i = 0; i < sof_match.size(); ++i) 
    {
      auto abs_indx = sof_match.position(i) + offset;
      offset += sof_match.position(i) + 5;
      mod_ind.push_back( abs_indx );
    }    
   
    cdata = sof_match.suffix(); 
  }
  mod_ind.push_back( fbin_len );

  //find function
  cdata = std::string( (char *) nvFCubinBase, fbin_len);
  std::string func_regex = "[^\\. ]+" + func_name + "[^\\.]+";
  std::regex fregex (func_regex);

  for(size_t i = 0; i < mod_ind.size()-1; i++)
  {
    std::string subst = cdata.substr(mod_ind[i], mod_ind[i+1] - mod_ind[i] );

    int ret = cuModuleLoadFatBinary(&cuModule, &((unsigned char *) nvFCubinBase)[mod_ind[i]]  );    

    if( ret == CUDA_SUCCESS ) 
    {
      if( std::regex_search(subst, sof_match, fregex) ) 
      {
        for (size_t j = 0; j < sof_match.size(); ++j) 
        {
          //std::cout << "Checking function : " << sof_match[j].str().data() << std::endl;  
          ret = cuModuleGetFunction(&khw, cuModule, sof_match[j].str().data() );

	  if( ret == CUDA_SUCCESS )
	  {
            //std::cout << "Successfuly found cubin for ..." <<  sof_match[j] <<  std::endl;
            *mangled_name  = sof_match[j];
	    *cubin_offset =  mod_ind[i];
            cuModuleUnload(cuModule);
            return 0;
	  } //end of loading function
	  
          cuModuleUnload(cuModule);
        } //end of loading module
      } //end of function mapping    
    } //end of regex_search
  } //end of for loop index
  
  return -1;
}	


void flash_cuda::_process_external_binary( const kernel_desc& kd, bool & found)
{
  //adding the module	
  _add_cuda_modules( kd._kernel_definition.value() );
  // add function to the library
  _add_cuda_function( kd._kernel_name.value(), found );
}

void flash_cuda::_add_cuda_function( std::string function_name, bool& found )
{
  std::cout << "entering " << __func__ << std::endl;

  //add a function and the module offset
  for(cuda_module& mod : _kernel_comps.modules )
  {
    size_t offset;
    std::string mangled_name;

    //find a valid cubin file within the fatcubin section
    unsigned char * data = mod.get_fbin_data<unsigned char>();      

    for( cuda_context& ctx : _kernel_comps.contexts)
    {
      //set current thread to context
      ctx.set_context_to_current();


      int err = _find_cubin_offset( data, mod.get_fbin_size(), 
			            function_name, &offset, &mangled_name);
      if(err == 0)
      {
	//create and add module to list
        bool added = mod.test_and_add( offset );

	//module is added
	if( added ) 
	{
	  CUfunction khw;
          err = cuModuleGetFunction(&khw, *mod.get_module(offset), mangled_name.c_str());
	  if( err == CUDA_SUCCESS)
	  {
            std::cout << "Found function entry point for : " << function_name << " : " << mangled_name <<std::endl;
	    std::string func_id = std::to_string( random_number() );
	    //add function
	    _kernel_comps.push_back(
	  		    cuda_function{
			      func_id,
			      function_name, mangled_name, khw
			    }
			  );
	    //create kernel_library entry
	    _kernel_lib.push_back( 
			     cuda_kernel{ func_id, function_name, mod.get_id(), ctx.get_id(), offset }
			    );

            found = true;
	  } //added function to functions
	} //add module to modules
      } //finding the cubin     
    } //context for loop
  } //module for loop

} 


void flash_cuda::_add_cuda_modules( std::string location )
{
  void * binary_mmap_ptr;
  Elf64_Ehdr elf_header;
  Elf64_Shdr header;
  std::string nvFatBinHdr = ".nv_fatbin";

  struct stat sb;
  size_t file_size=0;

  if( _kernel_comps.module_exists( location ) )
  {
    std::cout << "module already exists " << std::endl; 
    return; 
  }

  FILE *file = fopen( location.c_str(), "rb" );
  int fd     = fileno( file );
  
  auto elf_func = [&](auto func, auto ... parms)->bool {
    bool b = func( parms ...);
    fseek(file, 0, SEEK_SET);
    return b;
  };

  //check for a valid file descriptor and file pointer
  if( (fd >= 0) && file != nullptr )
  {
    if( fstat(fd, &sb ) != -1 )
    {
      file_size = sb.st_size;
      //mmap binaryt
      binary_mmap_ptr = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);

      if( binary_mmap_ptr != nullptr )
      {
        bool isElf = elf_func( elf_is_elf64, file);

        if( isElf )
        {
	  //get elf header
          elf_func( elf64_get_elf_header, file, &elf_header);

          bool contains_fatbinary = elf_func(elf64_get_section_header_by_name, 
	    		                     file, (const Elf64_Ehdr *) &elf_header, nvFatBinHdr.c_str(), &header);
	  if( contains_fatbinary ) 
	  {
            //Add the modules to the list
            _kernel_comps.push_back
	    ( 
              cuda_module{ location, file, file_size, binary_mmap_ptr, 
	                   fd, header.sh_offset, header.sh_size }
	    );
	  }
        }
      }
    }
  }
}

void flash_cuda::_retain_cuda_contexts()
{
  CUcontext ctx;
  CUdevice device;
  CUresult result;

  int deviceCount = 0;
  
  //get the number od devices
  cuDeviceGetCount( &deviceCount );  

  std::cout << "Found " << deviceCount << " devices" << std::endl;
  //get current context, and device
  for (int i : std::views::iota(0, deviceCount) )
  {
    //get device handles
    result = cuDeviceGet (&device, i);

    if( result == CUDA_SUCCESS)
    {
      //get contexts
      result = cuDevicePrimaryCtxRetain ( &ctx, device );

      if( result == CUDA_SUCCESS)
      {
	//push to compoements.context
        auto ctx_key = std::to_string( random_number() );
        _kernel_comps.push_back
	( 
          cuda_context{ ctx_key, 
			device, 
			ctx } 
	);

        //iontializing jobs table
	_job_count_per_ctx.insert({ctx_key, 0});

      }
    }
  }
}

void flash_cuda::_set_least_active_gpu_ctx()
{
  auto lowest = std::ranges::min_element(_job_count_per_ctx, 
  		                          {}, &std::map<std::string,ulong>::value_type::second);
  if( lowest != _job_count_per_ctx.end() )
    _kernel_comps.set_active_context( lowest->first );
  else std::cout << "Could not find gpu_ctx entry" << std::endl;
}

void flash_cuda::_assess_mem_buffers( ulong trans_id, ulong sa_id, std::vector<te_variable>& vars,
	                              std::vector<size_t> dims, te_attrs kAttrs )
{
  printf("entering _assess_mem_buffers tid = %llu, sid = %llu \n", trans_id, sa_id);
 
  cuda_context& cc = _kernel_comps.get_current_ctx();
   
  bool success = false;
  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
  for( auto& var : vars )
  {
    std::string buff_id = var.get_mem_id();
    
    if( !_mem_registry.count(buff_id) )
    {
      allocate_buffer( var, success );
      printf("Allocating new Buffer %s : ... : %i\n", buff_id.c_str(), success );
    }
    else 
    {
      printf("Buffer %s is already registered\n", buff_id.c_str() );
      success = true;
    }

    if( success )
    {
      mem_detail& md = _mem_registry.at( buff_id );

      md.reconcile_memstate( trans_id, sa_id, cc, var.parm_attr.dir );
    }
  }

  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
  printf("Entering table creation...\n");
  if( index_table::table_required( kAttrs )  )
  {
    std::string table_id = index_table::preview_table_id(dims);
    if( !_mem_registry.count( table_id) )
    {
      printf("Allocating new Buffer %s : ... : %i\n", table_id.c_str(), success );
      _mem_registry.emplace(std::piecewise_construct,
                            std::forward_as_tuple(table_id), 
		            std::forward_as_tuple(dims, kAttrs) );
      
    }
    else
    {
      printf("Buffer %s is already registered\n", table_id.c_str() );
      success = true;
      
    }

    printf("fetching table : %s...\n", table_id.c_str() );
    mem_detail& md = _mem_registry.at( table_id );
    printf("reconciling table : %s...\n", table_id.c_str() );
    md.reconcile_memstate( trans_id, sa_id, cc, DIRECTION::IN );
    printf("completed table reconcil : %s...\n", table_id.c_str() );

  }
  printf("completed table creation...\n");
  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////

}

void flash_cuda::_writeback_to_host( ulong tid, te_variable& arg)
{
  //TBD

}

void flash_cuda::_deallocate_buffer(ulong, DIRECTION dir, te_variable& arg)
{
  //TBD


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void mem_detail::allocate( std::string tid, cuda_context ctx, DIRECTION dir, bool init )
{
  printf("    mem_detail::allocate tid = %s \n", tid.c_str() );
 
  if( !tid_exists(tid) )
  { 
    CUdeviceptr dptr;
    auto& mstate = add_memstate( tid, ctx, dir );

    cuMemAlloc ( &dptr, sz );
    mstate.dptr = dptr;
    printf("------------>Allocating sz = %llu, %p<--------------\n", sz, dptr );

    if( (dir == DIRECTION::IN) && init )
    {
      printf("Transferringg from host to device...\n");
      host_to_device( dptr );
      printf("Transferred from host to device...\n");
    }
  }
  else std::cout<< "Buffer already allocated" << std::endl;
}

void mem_detail::deallocate( std::string tid, bool transfer_back )
{
  auto mstate = get_mstate( tid );
  if( (mstate.dir != DIRECTION::IN) && 
       !is_flash() && transfer_back ) 
  {
    if( mstate.dptr ) device_to_host( mstate.dptr.value() );
    else std::cout << "No device ptr to transfer data (Dealloc)" << std::endl;   

  } else std::cout << "Cannot deallocate buffer" << std::endl;

  if( mstate.dptr ) cuMemFree( mstate.dptr.value() );
  else std::cout << "No device ptr to Dealloc" << std::endl;   

  mstate.dptr.reset(); 
}

void mem_detail::deallocate_all( bool transfer_back )
{
  auto dealloc = std::bind( &mem_detail::deallocate, this, std::placeholders::_1, transfer_back );

  //deallocate all transactions for this buffer
  std::ranges::for_each( _mstates, dealloc, &mem_state::tid );
}

void mem_detail::transfer( std::string target_tid, cuda_context target_ctx, DIRECTION target_dir, bool transfer_data )
{
  mem_state mstate = _mstates.back();
  std::string src_ctx_key = mstate.ctx->get_id();

  if( src_ctx_key != target_ctx.get_id() )
  {
    allocate( target_tid, target_ctx, target_dir, false);
    mem_state tmstate = _mstates.back();
    device_to_device( mstate.ctx.value().cuContext,  
                      mstate.dptr.value(),
                      tmstate.ctx.value().cuContext, 
                      tmstate.dptr.value() );     
  } 
  else
  {
    //ONLY need to update  tid, and dir
    mstate.tid = target_tid;
    mstate.dir = (mstate.dir==DIRECTION::IN)?target_dir:mstate.dir;
    mstate.ctx = target_ctx;
  }
  
}

void mem_detail::host_to_device( CUdeviceptr dptr )
{
  printf("host_to_device: device_ptr =%p, host_data=%p, sz=%llu\n",
	  dptr, host_data, sz );

  cuMemcpyHtoD ( dptr, host_data, sz );
}

void mem_detail::device_to_host( CUdeviceptr dptr )
{
  cuMemcpyDtoH ( host_data, dptr, sz );
}

void mem_detail::device_to_device( CUcontext src_ctx, CUdeviceptr src_ptr, CUcontext dst_ctx, CUdeviceptr dst_ptr)
{
  cuMemcpyPeer ( dst_ptr, dst_ctx, src_ptr, src_ctx, sz );
}

void mem_detail::reconcile_memstate( ulong tid, ulong sa_id, cuda_context ctx, DIRECTION dir )
{

  printf("  Entering reconcile_memstate tid = %llu, sa_id = %llu \n", tid, sa_id);
  std::string ctx_key = ctx.get_id();
  auto tid_str = std::to_string( tid );
  printf("    Mark 1\n");
  
  //memory state on same device (sd)
  auto ms_sd = std::ranges::find( _mstates, ctx_key, 
                                  &decltype(_mstates)::value_type::get_cid);
  
  printf("    Mark 2\n");
  //memory state on different device (dd) 
  auto ms_dd = std::ranges::find_if( _mstates, unary_diff{ctx_key}, 
                                     &decltype(_mstates)::value_type::get_cid);
 
  printf("    Mark 3\n");
  auto key = std::make_pair( tid, sa_id );

  //Buffer on the same device
  if( ms_sd != _mstates.end() ){
    printf("    Mark 4\n");
    ms_sd->dir = dir;
    ms_sd->tid = tid_str;

    if( !ms_sd->in_use || (ms_dd->dir == DIRECTION::IN) ){
     std::cout << "Reassigning memstate" << std::endl;  
    }
    else 
    {
      printf("    Mark 5\n");
      auto states = std::make_pair(*ms_sd, *ms_sd );

      _pending_transfers.emplace(key, states );
    }

  }
  //Buffer different devices
  else if( ms_dd != _mstates.end() ){
    printf("    Mark 6\n");
    allocate( tid_str, ctx, dir, false );
    auto ms = get_memstate( std::to_string(tid) );

    if( !ms_dd->in_use || (ms_dd->dir == DIRECTION::IN) ) 
      device_to_device(ms_dd->ctx->cuContext,
                       ms_dd->dptr.value(),
                       ms.ctx->cuContext,
                       ms.dptr.value() );
    else
    {
      printf("    Mark 7\n");
      auto states = std::make_pair(*ms_dd, ms );

      _pending_transfers.emplace(key, states );
    }

  }
  //no buffers anywhere
  else{
    //buffer exists in this context/device
    printf("    Mark 8\n");
    allocate( tid_str, ctx, dir );
  }

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////index_table_intf//////////////////////////////////////////////////////////
/////////////////-/////////////////////////////////////////////////////////////////////////////////////////////////////
/*std::vector<size_t> index_table::generate_offsets( std::vector<size_t>, te_attr )
{

  return {};
}
*/
std::string index_table::preview_table_id( const std::vector<size_t>& table_dims )
{

  auto underscore = [](std::string id, size_t sz) 
  {
    return std::move(id) + '_' + std::to_string(sz);
  };
	 
  return std::accumulate(std::begin(table_dims), 
                         std::end(table_dims),
			 std::string("table"),
			 underscore); // start with first element
  
}


index_table::index_table( std::vector<size_t> table_dims, 
                          te_attrs kernel_attrs )
{
  _sz =  std::accumulate(table_dims.begin(), 
		         table_dims.end(), 1, std::multiplies<size_t>() );
  _dims = table_dims;
  _hptr = (size_t *) malloc( _sz*sizeof(size_t) );
  _id = preview_table_id(table_dims);

  _fill_host_memory(table_dims);

  _generic_offsets.push_back(0);

  if( !kernel_attrs.empty() ) 
    reorder ( kernel_attrs );


}
//This is a list ofindexes
std::vector< std::vector<size_t> >
index_table::get_dim_arrays()
{
  //this functionre retuns for each subdispatch returns the new dimsension
  using ret_valtype = std::vector<size_t>;
  std::vector<ret_valtype> out;

  printf("index_table::get_dim_arrays() : %llu \n", 
	 _generic_offsets.size() );

  if( _generic_offsets.size() == 1)
  {
    out.push_back(_dims);
    return out;
  }
  
  size_t num_parts = 1;

  for(uint i=_split_dim-1; i < _dims.size(); i++)
  {
    printf("index_table::get_dim_arrays() : _dims.size = %i\n", _dims.size() );
    num_parts *= _dims.at(i);
  }

  printf("index_table::get_dim_arrays() : num_part : %llu\n", num_parts );
  for(uint i=0; i < num_parts; i++)
  {
    ret_valtype entry;
    for(uint j=0; j < _dims.size(); j++)
      entry.emplace_back(j<(_split_dim-1)?_dims.at(j):1 );
	
    out.push_back( std::move(entry) );
  } 

  return out;
}

void 
index_table::_calc_generic_offsets( )
{
  size_t num_parts = 1, element_offset=1;

  if( _split_dim > _dims.size() )
  {
    printf("_split_dim > _dim.size(), skipping offset calc\n");
    _generic_offsets.push_back(0);
    return; 
  }

  for(uint i=_split_dim-1; i < _dims.size(); i++)
    num_parts *= _dims.at(i);
 
  for(uint i=0; i < _split_dim-1; i++)
    element_offset *= _dims.at(i);

  //element_offset *= _dims.size();
  printf("-- element_offset = %i\n", element_offset);

  auto byte_offset = element_offset * sizeof(size_t);
 
  for(size_t i=1; i < num_parts; i++){
     if (i < 10) printf("---Offset(parts=%i) = %i\n", i, byte_offset*i);
    _generic_offsets.push_back(byte_offset*i);
  }
  printf("finishing _calc_generic_offset...\n");
}	


/*operator index_table::te_variable() const
{
  auto tv = te_variable{
              _hptr,
              sizeof(size_t),
              _sz,
              ParmAttr{true, false, false, false, false, DIRECTION::IN},
              {},{}, _id

            };

  return tv;

}*/


void index_table::_fill_host_memory( std::vector<ulong> dims)
{
  for(ulong ui=0; ui < _sz/dims.size(); ui++)
  {
    for(uint dim=0; dim < dims.size(); dim++)
    {
      _hptr[ui+dim] = ui % dims[dim];
    }  

  }
}

void index_table::reorder( te_attrs kattrs )
{
  if( kattrs.size() == 0 ) return;

  if(kattrs.size() == 1 )
  {

    printf("Getting split dim...: %i\n", kattrs.front().dims.size() );

    _split_dim = kattrs.front().dims.at(0);
    printf("Got split dim... : %i\n", _split_dim);

    printf("kattrs.front() = %s \n", 
           kattrs.front().id == KATTR_SORTBY_ID?"SORTBY":"GROUPBY" );

    if( kattrs.front().id == KATTR_SORTBY_ID )
      _sort_host_memory( kattrs.front().dims );
    else if( kattrs.front().id == KATTR_GROUPBY_ID )
      _group_host_memory( kattrs.front().dims );
  } else std::cout << "multiple attributes unsupported" << std::endl;

  _calc_generic_offsets();
}

void index_table::_sort_host_memory( std::vector<size_t> dims )
{
  printf("Sorting table ...\n");
}

void index_table::_group_host_memory( std::vector<size_t> dims )
{
  printf("Group table ...\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////index_table_intf//////////////////////////////////////////////////////////
/////////////////-/////////////////////////////////////////////////////////////////////////////////////////////////////
cuda_index_table::cuda_index_table(const CUdeviceptr& devptr, 
		                   const mem_detail& md )
: _dptr( devptr ), _md(md)
{
  _convert( );
}

void
cuda_index_table::_convert( )
{
  auto o_table = _md.get_table();
  
  if( !o_table )
    std::cout <<"mem detail is not a table ..." << std::endl;
  else
  {
    auto offsets = o_table->get_generic_offset();
    std::ranges::for_each(offsets, [&](auto offset)
    {
      _segments.push_back(_dptr + offset);
    });
  }
}

std::vector<CUdeviceptr>
cuda_index_table::range()
{
  return _segments;
}

std::map<CUdeviceptr, std::vector<size_t> > 
cuda_index_table::get_dim_parts()
{
  auto o_table = _md.get_table();
  std::map<CUdeviceptr, 
	     std::vector<size_t> > out;
  
  if( !o_table )
    std::cout <<"mem detail is not a table ..." << std::endl;
  else
  {
    auto new_dims = o_table->get_dim_arrays();
    auto dev_ptrs = range();
    int i =0;
    std::ranges::transform (dev_ptrs, new_dims, std::begin(new_dims),
    [&](auto dev_ptr, auto dims)
    {
      if( i < 10 ) printf( "----dev_ptr = %llu, dim_array_size = %llu \n", dev_ptr, new_dims.size() );
      out.insert({dev_ptr, dims});
      i++;
      return dims;
    } );

  }

  return out;
}
