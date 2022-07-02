#include <flash_runtime/flashrt.h>
#include <flash_runtime/transaction_interface.h>
#include <ranges>
#include <algorithm>
#include <functional>
#include <future>

//std::shared_ptr<flash_rt> flash_rt::_global_ptr;

//use for kernels function
extern size_t EXPORT get_indices( int ind);


std::shared_ptr<flash_rt> EXPORT flash_rt::get_runtime( std::string runtime_lookup )
{

  if( _global_ptr )
  {
    std::cout << "_global_ptr already exists" << std::endl;
    return _global_ptr->_customize_runtime(runtime_lookup);
  }
  else
  {
    std::cout << "Creating new global pointer " << std::endl;
    _global_ptr = std::shared_ptr<flash_rt>( new flash_rt( "ALL" ) );

    return _global_ptr->_customize_runtime(runtime_lookup);
  }
}

std::shared_ptr<flash_rt> flash_rt::_customize_runtime( std::string rt_key )
{
  std::cout << "customizing runtime for " << rt_key <<  std::endl;
  return _rtrs_tracker.get_create_runtime( rt_key );
}


EXPORT flash_rt::flash_rt( std::string lookup)
{

  if( !lookup.empty() && lookup != "ALL" )
  {
    std::cout<< "creating a flashrt with " << lookup << std::endl;
    _runtime_key = lookup;
    _backend = FlashableRuntimeFactory::Create( lookup );
    //get pointer to backend runtime
    _runtime_ptr = _backend.value()();
    //_runtime_ptr->set_trans_intf( std::shared_ptr<transaction_interface>( &_trans_intf ) );
    _runtime_ptr->set_trans_intf( _trans_intf  );
    std::cout << "Ctor'ing flash_rt...." << std::endl;

    std::cout << _backend->get_description() << std::endl; 
  }
  else
  {
     std::cout << "Returning nullptr" << std::endl;
    _runtime_key = g_NoRuntime;
    //this indicates to the system that it should
    //choose the runtime
    _runtime_ptr = nullptr;
    //populating  resource tracker
    for(auto rtk : FlashableRuntimeFactory::List() ) 
    {
      std::cout << "Registering runtime : " << rtk << std::endl;
      _rtrs_tracker.get_create_runtime( rtk );
    }
       
  }
}

std::string flash_rt::_recommend_runtime( const std::string& kernel_name,
                                          const std::vector<te_variable>& kernel_args )
{
  std::cout << __func__ << " Started " << std::endl;
  std::cout << "  Looking for kernel name " << kernel_name <<  std::endl; 
  auto kernel_rts = _rtrs_tracker.get_all_runtimes_by( kernel_name );
  std::vector<std::string> kernel_rtks, buffer_rtks;

  std::cout << "kernel_rts size = " << kernel_rts.size() << std::endl;

  std::ranges::transform( kernel_rts, std::back_inserter(kernel_rtks),
                          [](auto kernel_rt)
  {
    auto rt =  kernel_rt->get_runtime_key();
    std::cout << "Found runtime : " << rt.value_or(g_NoRuntime) << std::endl;
    return rt.value_or(g_NoRuntime);

  } );

  std::cout << " " << "got list of keys size = " << kernel_rtks.size() << std::endl;
  for(auto rk : kernel_rtks ) std::cout << "    " << rk << std::endl;
  /////////////////////////////////////////////////////////////////////////////////////////////
  auto fmem_pred = [](auto arg) -> bool
  {  
    return arg.is_flash_mem();
  };

  auto fmems = kernel_args | std::views::filter(fmem_pred);

  std::ranges::transform( fmems, std::back_inserter(buffer_rtks),
                          [&](auto fmem_arg ) ->std::string
  {

    auto fmem_id = fmem_arg.get_fmem_id();
    //auto runtime = _rtrs_tracker.get_runtime_by_fmem( std::to_string( fmem_id ) );
    auto runtime = _rtrs_tracker.get_runtime_by_mem( fmem_id );
    if( runtime )
      return *runtime->get_runtime_key();
    else
      return g_NoAlloc;

  } );
  std::cout << " " << "finished analyzing buffers, size = " << buffer_rtks.size() << std::endl;
  ///////////////////////////////////////////////////////////////////////////////////////////// 
  auto none_alloc    = std::ranges::all_of( buffer_rtks, unary_equals{ g_NoAlloc } );  
  std::cout << "  Mark 1" << std::endl;
  auto is_colocated  = std::ranges::all_of( buffer_rtks, unary_equals{ kernel_rtks[0] } );  
  std::cout << "  Mark 2" << std::endl;

  if( none_alloc ) 
    std::cout << "None of the flash_memory buffers allocated! : " << kernel_rtks[0] << std::endl;
  else if( is_colocated ) 
    std::cout << "Kernel and buffers are colocated in "<< kernel_rtks[0] << std::endl; 
  else
    std::cout << "Mix of runtimes found!" << std::endl; 

  return kernel_rtks[0];
}

std::function<int()>
flash_rt::_manage_buffers( std::string tid, std::string rtk, std::vector<te_variable>& kernel_args )
{
  std::cout << __func__ << " Mark 0" <<std::endl;

  auto exec_rt = _rtrs_tracker.get_create_runtime( rtk );
  std::function<int()> out;

  std::cout << __func__ << " Mark 1" <<std::endl;

  int i =0;
  std::vector<int> indxs;
  for( auto& mem : kernel_args )
  {
    std::cout <<" mem_id = " << mem.get_mem_id() << std::endl;
    std::cout <<" "<< __func__ << " Mark Arg " << i << " : started" << std::endl;
    auto buffer_rt  = _rtrs_tracker.get_runtime_by_mem( mem.get_mem_id() );
    std::cout << "    " << __func__ << "Mark 0" << " : inner" << std::endl;
    auto buffer_rtk = buffer_rt?buffer_rt->get_runtime_key():g_NoAlloc;
    std::cout << "    " << __func__ << "Mark 1" << " : inner" << std::endl;

    if( rtk == buffer_rtk ) 
      std::cout << "Buffer " << mem.get_mem_id() << " already allocated" << std::endl;
    else if ( buffer_rtk == g_NoAlloc )
    {
      bool res = exec_rt->allocate_buffer(mem);
      if( res ) _rtrs_tracker.register_mem( tid, rtk, mem );
    }
    else if( rtk != buffer_rtk ) indxs.push_back( i );

    std::cout <<" "<< __func__ << " Mark Arg " << i << " : end" << std::endl;
 
    i++;
  }

  std::cout << __func__ << " Mark 2" <<std::endl;
  out = [this, rtk, indxs, kernel_args]()-> int
  {

    for( int i : indxs )
    {
      this->_rtrs_tracker.transfer_buffers( rtk, {}, {kernel_args.at(i)} ); 
    }
    return 0;
  };

  std::cout << __func__ << " Mark 3" <<std::endl;
  return out;
}

status EXPORT flash_rt::execute(runtime_vars rt_vars,  uint num_of_inputs, 
                         std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms, v_options& opt)
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  //need to store all arguments for later processing through process_transaction
  auto[trans_id, suba_id] = rt_vars.get_ids();
  //create a subactions
  std::ranges::for_each(kernel_args, [](auto arg)
  {
    if( arg.data == nullptr) std::cout << "flashrt::kernelarg == nullptr" << std::endl;

  });
 
  std::string rtk = _runtime_key.value_or(g_NoRuntime);

  if( rtk == g_NoRuntime )
  {
    auto kname_ovr   = rt_vars.get_kname_ovr();
    auto kernel_name = kname_ovr.value_or( rt_vars.get_lookup() );

    rtk = _recommend_runtime( kernel_name, kernel_args);
  }
  
  std::cout << "runtime selected : " << rtk << std::endl;
  auto sa = subaction{ suba_id, rtk, num_of_inputs, rt_vars, kernel_args, 
                       exec_parms, opt };

  //add transactio and subactionsa
  auto& sa_ref = _trans_intf.add_sa2ta( trans_id, std::move(sa) );
  
  std::cout << __func__ << " Mark 6" << std::endl;
  auto deferred_transfer = 
       _manage_buffers( std::to_string(trans_id), rtk, kernel_args );
  //check predecessor conditionals
  //
  std::cout << __func__ << " Mark 7" << std::endl;
  auto [dep_pred, succ_pred] = _trans_intf(trans_id).get_pred(suba_id);

  std::cout << __func__ << " Mark 8" << std::endl;
  auto pred = [dep_pred, deferred_transfer]()->int
  {
    std::cout << "calling dep_pred...\n";
    dep_pred();
    std::cout << "calling deferred_transfer...\n";
    deferred_transfer();
    return 0;
  };

  std::cout << __func__ << " Mark 9" << std::endl;
  sa_ref.set_preds( std::move(pred), std::move(succ_pred) );


  return {};
}

status EXPORT flash_rt::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                          std::string kernel_names[], std::optional<std::string> kname_ovrs[],
                                          std::optional<std::string> inputs[] ) 
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;

  std::vector<kernel_desc> kernel_inputs;
  std::string rtk = _runtime_key.value_or(g_NoRuntime);

  auto pack_data = [&](int index)-> kernel_desc
  {
    std::cout << "base kernel_name = " << kernel_names[index] << ", override = ";

    if( kname_ovrs[index] ) std::cout << kname_ovrs[index].value() << std::endl;
    else std::cout << "No Override" << std::endl; 

    if( !kname_ovrs[index] )
      return kernel_desc{kernel_types[index], kernel_names[index], 
                         kernel_names[index], inputs[index]};
    else
      return kernel_desc{kernel_types[index], kernel_names[index], 
                         kname_ovrs[index],   inputs[index]};
   
  };

  auto k_exists = std::bind(&runtimes_resource_tracker::kernel_exists, 
                            &_rtrs_tracker, rtk, std::placeholders::_1); 

  //check if thier is a runtime existsa
  auto kernels = std::views::iota( (size_t) 0, num_kernels ) | 
                 std::views::transform(pack_data) |
                 std::views::filter( std::not_fn(k_exists) );

  //pack the inputs into
  std::ranges::for_each(kernels, [&](auto input)
  { 
    std::cout << "  Pushing data into kernel_inputs" << std::endl;
    kernel_inputs.push_back(input);

  } ); 
 
  
  if( !kernel_inputs.empty() ) _try_register_kernel( kernel_inputs, rtk );

  std::cout << "completed flash_rt::" << __func__ <<  std::endl;
  return {}; 
}

status flash_rt::allocate_buffer( te_variable& )
{

  return status{};
}

ulong EXPORT flash_rt::create_transaction()
{
  ulong tid = random_number();

  return tid;
}

status EXPORT flash_rt::process_transaction( ulong tid )
{
  std::cout << "calling flash_rt::" << __func__ <<"("<<tid <<")" << std::endl;
  ///////////////////////////////////////////////////////////
  std::vector<std::future<status> > statuses;
  strings rtks;
 
  auto [b_iter, e_iter] = _trans_intf.get_transaction(tid);

  _trans_intf.demarc_boundaries(tid);
  
  std::cout << "--Got transactions" << std::endl;
  if( !_runtime_ptr ) std::runtime_error("No backend selected/available!");

  if( auto dist = std::distance(b_iter, e_iter); dist != 0  )
  {
    std::cout << "--- dist != 0" << std::endl;
    //check if thier is a runtime exists
    //std::vector<std::reference_wrapper<subaction> > pipeline(b_iter, e_iter);
    std::vector<std::reference_wrapper<subaction> > pipeline;

    std::transform( b_iter, e_iter, std::back_inserter( pipeline ), []( auto& subacts )
    {
      return std::ref(subacts.second); 
    } );

    std::cout << "---- transformation complete" << std::endl;
    //sort by subaction id
    std::ranges::sort( pipeline, {}, &subaction::subaction_id);

    std::cout << "******************************************************************************" << std::endl;
    std::cout << "****************************Execution Section*********************************" << std::endl;
    std::cout << "******************************************************************************" << std::endl;
    //submit subactions
    std::for_each( pipeline.begin(), pipeline.end(), [&](subaction& stage)
    {
      std::cout << "------------------------------------------------" << std::endl;
      auto[num_of_inputs, rt_vars, kernel_args, exec_parms,
           pre_pred, post_pred ] = stage.input_vars();

      auto sa_id = stage.get_saId();
      auto opts  = stage.get_options();
      auto ktype = rt_vars.get_ktype();
      auto base_kname = rt_vars.get_lookup();
      auto kname_ovr  = rt_vars.get_kname_ovr();
      auto kname_impl = rt_vars.get_kimpl();
      //register_kernels( 1, &ktype, &base_kname, &kname_ovr, &kname_impl ); 
     
      auto runtime  = _rtrs_tracker.get_create_runtime( stage.get_rtk() ); 
      rtks.push_back( stage.get_rtk() );                    
      
      std::cout << "executing " << kname_ovr.value_or(base_kname) << ", status = ";
      //statuses.push_back( runtime->execute(rt_vars, num_of_inputs, kernel_args, exec_parms, opts ) );
      auto exec_future = std::async(std::launch::async, 
                         [=]{
                           printf("\nFLASH Executing thread_id= %i tid = %llu, sa_id = %llu",
                                  std::hash<std::thread::id>{}(std::this_thread::get_id()),
                                  tid, sa_id );  
                           return runtime->execute(tid, sa_id ); 
                         });
      statuses.push_back( std::move(exec_future)  );
      std::cout << "In progress on "<< stage.get_rtk() << " ..." << std::endl;
      std::cout << "------------------------------------------------" << std::endl;

    });
    /*std::cout << "******************************************************************************" << std::endl;
    std::cout << "****************************Wait for all Executes to finished*****************" << std::endl;
    std::cout << "******************************************************************************" << std::endl;
    auto exec_wait = std::bind(&std::future<status>::wait, std::placeholders::_1);
    std::ranges::for_each( statuses, exec_wait );  */ 

    std::cout << "******************************************************************************" << std::endl;
    std::cout << "********************************Wait Section**********************************" << std::endl;
    std::cout << "******************************************************************************" << std::endl;
    ///wait for work to complete
    auto completed = statuses | std::views::transform([&](auto& stat_future) 
    {
      status stat;

      if( !stat_future.valid() ) 
      {
        printf("Future is not valid...\n");
        return std::future<status>();
      }
     
      stat = stat_future.get();
 
      if ( !stat ) {
        printf("failed to execute....\n");
        return std::future<status>();
      }

      if( stat.work_id )
      {
        auto wid = stat.work_id.value();
        std::cout << "waiting on " << wid << " to complete..." << std::endl;
        auto runtime  = _rtrs_tracker.get_create_runtime( rtks.front() ); 
        
        auto ret = std::async(std::launch::async,
		   [&]{
                     auto ret = runtime->wait( wid );
                     rtks.erase( rtks.begin() );
	             return ret;
		   });

	ret.wait();
        std::cout << "launching wait " << wid << std::endl;
        return ret;
      }
      else 
      {
        std::cout << "-Could not find wid-" << std::endl;
        return std::future<status>();
      }

    } ) | std::views::transform([&](auto&& wait_future)
	 {
	   ////waits for wait to complete
	   if(wait_future.valid() )
	   {
             std::cout << "issuing wait.... is_valid : " <<wait_future.valid() << std::endl;
	     wait_future.wait();
             std::cout << "waiting.... " << std::endl;
             auto ret = wait_future.get();
	     if( ret.work_id )
	     {
               auto wid = ret.work_id.value();
               std::cout << "completed .... " << wid << std::endl;
	       return status{};
	       //return ret;
	     }
	     else 
	     {
	       printf("Invalid return for wait future...\n");
	       return status{-1};
	     }

	   }
	   else return status {-1};
    });

    std::cout << "----- starting pipeline" << std::endl;
    auto all_complete = std::ranges::all_of( completed, unary_equals{0}, &status::err );
    std::cout << "----- pipeline complete" << std::endl;
   
    if( all_complete ) std::cout << "successfully completed tid = " << tid << std::endl;
    else std::cout << "failed to complete tid = " << tid << std::endl; 
    
  }
  else
  {
    std::cout << "Could not locate transaction " << tid << std::endl;
  }

  std::cout << "******************************************************************************" << std::endl;
  std::cout << "********************************Wait Section Complete*************************" << std::endl;
  std::cout << "******************************************************************************" << std::endl;

  return {};
}

void flash_rt::_try_register_kernel(std::vector<kernel_desc>& kds, std::optional<std::string> rtk)
{

  std::vector<bool> successes;
  std::string rkey;

  std::cout << "calling flash_rt::" << __func__ << std::endl;

  auto
  update_loc = [&rkey](auto& kd) 
  {
    if( !kd.get_kernel_def() || (kd.get_kernel_def() && (kd.get_kernel_def().value() == g_NotImpl)) )
    {
      auto base_name = kd.get_base_kname();
      auto base_loc  = _rtrs_tracker.get_base_loc( rkey, base_name );
      kd.set_kernel_def( base_loc );

      std::cout << "Overriding location for " << kd.get_kernel_name() << 
                " to location of " << base_name << " @ " << base_loc << std::endl; 
    }  
  };

  auto
  register_func = [&](auto rt)->void
  {
    rkey = rt->get_runtime_key().value();
    std::cout << "Trying to Register the following kernels in " << rkey << " : " << 
                 (bool) rt->get_backend() <<  std::endl;

    std::ranges::for_each( kds, update_loc);

    rt->get_backend()->register_kernels( kds, successes );

 
    //go through each success registration and add it to the registry
    std::ranges::transform(kds, successes, std::back_inserter(successes),
    [&](const kernel_desc& kd, const bool& success)
    {
      std::cout << "  " << kd.get_kernel_name() << ", status =  " << success << std::endl;
      if( success ) _rtrs_tracker.register_kernel(rkey, kd );
      return false;
    } );
    
  };

  if( !rtk || ( rtk && (rtk.value() == g_NoRuntime) ) )
  {
    std::cout << "  Testing all runtimes" << std::endl;
    for( auto runtime : _rtrs_tracker.get_all_runtimes() ) 
      register_func( runtime );
  }
  else 
  {
    std::cout << "  Testing single runtime " << (bool) rtk <<  std::endl;
    auto runtime  = _rtrs_tracker.get_create_runtime( rtk.value() ); 
    register_func( runtime );
  }

}



/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
runtimes_resource_tracker::shared_flash_runtime
runtimes_resource_tracker::get_create_runtime(std::string rtk )
{
  if( _runtime_ptrs.count( rtk ) == 0 )
  {
    std::cout << "Registering runtime with the resource tracker...." << rtk << std::endl;
    //auto srtp = std::make_shared<flash_rt>(rtk);
    auto srtp = std::shared_ptr<flash_rt>(new flash_rt( rtk) );
    _runtime_ptrs[rtk] = srtp;
  }

  std::cout << "Returning a runtime...." << rtk << std::endl;
  return _runtime_ptrs[rtk];
}

bool
runtimes_resource_tracker::runtime_exists( std::string rtk )
{
  return _runtime_ptrs.count(rtk) > 0;
}


runtimes_resource_tracker::shared_flash_runtime
runtimes_resource_tracker::get_runtime_by_kname( std::string kname )
{
  
  runtimes_resource_tracker::shared_flash_runtime rt_sptr;

  rt_sptr = _get_runtime_by( kname );

  return rt_sptr;
}



runtimes_resource_tracker::shared_flash_runtime
runtimes_resource_tracker::get_runtime_by_mem( std::string mem_id )
{
  std::cout << __func__ << " Mark 0" << std::endl;
  auto rt_sptr = _get_runtime_by( mem_id );
  return rt_sptr;
}

runtimes_resource_tracker::shared_flash_runtime
runtimes_resource_tracker::_get_runtime_by( std::string id )
{
  std::cout << __func__ << " Mark 0" << std::endl;
  auto rts = get_all_runtimes_by( id );
 
  if( rts.size() == 0 ) {
    std::cout << "Buffer id = " << id << " not registered " << std::endl;
    return nullptr;
  }

  return rts[0];
}

void
runtimes_resource_tracker::register_mem( std::string tid, std::string rtk, const te_variable& mem )
{
  
  summary_flash_mem sfm;
  sfm.tid       = tid; 
  //sfm.id        = std::to_string( fmem.get_fmem_id() );
  sfm.id        = mem.get_mem_id();
  sfm.type_size = mem.type_size;
  sfm.vec_size  = mem.vec_size;
  sfm.base_addr = mem.data;

  _resources.emplace( rtk, sfm );
  

}

auto 
runtimes_resource_tracker::get_all_runtimes_by( const std::string& id )
-> std::vector<shared_flash_runtime>
{

  std::vector<shared_flash_runtime> out;

  auto no_all = [](auto rt)
  {
    return !(rt.first == g_NoRuntime);
  };

  if( id.empty() ) 
    return (std::ranges::copy(_runtime_ptrs | std::views::filter(no_all) | std::views::values, 
                              std::back_inserter(out) ), out);

  std::cout << __func__ << " Mark 0 : resournce size() = " << _resources.size() <<  std::endl;
  for(auto[rtk, summary_obj] : _resources )
  {
    std::cout <<"  "<< __func__ << " Mark 0" << std::endl;
    std::visit([&](auto obj)
    {
      if constexpr (std::is_same_v<decltype(obj), summary_kernel > )
      {
        std::cout << "obj.kernel_name = " << obj.kernel_name << std::endl;
        if( obj.kernel_name == id )
          out.push_back( get_create_runtime( rtk ) ); 
      }
      else if constexpr (std::is_same_v<decltype(obj), summary_flash_mem> )
      {
        std::cout << "obj.id = " << obj.id << std::endl;
        if( obj.id == id )
          out.push_back( get_create_runtime( rtk ) ); 
      }
    }, summary_obj);
    //break if we found a runtime
  }

  std::cout << __func__ << " : end" << std::endl;
  std::cout << __func__ << " : out.size() = " << out.size() << std::endl;
 
 
  return out;
}

void
runtimes_resource_tracker::register_kernel( std::string rtk, const kernel_desc& kd )
{
  auto base_kname = kd.get_base_kname();             //kernel name
  auto base_loc   = get_base_loc( rtk, base_kname);  //location
  auto ovr_kname  = kd.get_ovr_kname();            //optional kernel name
  auto ovr_loc    = kd.get_kernel_def();             //optional location

  std::string kname = ovr_kname?ovr_kname.value():base_kname;
  std::string loc   = ovr_loc?ovr_loc.value():base_loc;

  std::cout << "entering runtimes_resource_tracker::" << __func__ << std::endl;
  std::cout << "Registering " << kname << " @ " << loc << std::endl;

  summary_kernel sk{ kname, loc };

  _resources.emplace(rtk, sk);
}

std::string runtimes_resource_tracker::get_base_loc( std::string rtk, std::string kname)
{
  auto rt_resources = _resources.equal_range( rtk );
  std::optional<std::string> location;

  for(auto iter = rt_resources.first; iter != rt_resources.second; iter++)
  {
    std::visit( [&location, kname](auto res)
    {
      if constexpr( std::is_same_v<summary_kernel, decltype(res)> )
      {
        if( res.kernel_name == kname ) 
          location = res.kernel_location;
      }

    }, iter->second );

    if( location ) break;
  } 

  if( location ) return location.value();
  else std::logic_error("Error: no base location for function" + kname); 

  return "";
}


bool runtimes_resource_tracker::kernel_exists( std::string rtk, const kernel_desc & kd)
{
  std::cout << "entering " <<__func__ << " " << rtk<< std::endl;

  auto [stIt, endIt] = _resources.equal_range( rtk );

  if( rtk == g_NoRuntime )
  {
    stIt  = std::begin(_resources);
    endIt = std::end(_resources);
  }

  std::string kname  = kd.get_kernel_name();
  std::string loc    = kd.get_kernel_def().value_or(g_NotImpl);

  std::cout << "res.size() = " << _resources.size() << ", " << "kname = " << kname << ", loc = " << loc << std::endl;
  if( stIt == endIt ) std::cout << "No resources in resources list" << std::endl;
  auto ret = std::ranges::any_of( stIt, endIt, [&]( auto var_res )
  {
    bool ret = false;

    std::visit( [&](auto res)
    {
      if constexpr( std::is_same_v<summary_kernel, decltype(res)> )
      {
        std::cout << "kname : " << kname <<" == " << res.kernel_name <<" , " <<
                  "loc : " << loc << " == " << res.kernel_location << std::endl;

        ret = ( ( kname == res.kernel_name) && 
              ( (loc == res.kernel_location) || (loc == g_NotImpl) ) );
        if(ret) std::cout << "  Found kernel in resources : " << kname << " @ " << loc << std::endl;
      }
      else
      {
        std::cout << "  Other type found! " << std::endl;
      }

    }, var_res.second );
   

    return ret;

  } );

  return ret;
}

bool runtimes_resource_tracker::transfer_buffers( o_string dst_rtk, o_string src_rtk, te_variables buffs )
{
  //TBD dont foret to deallocate from src runtime.
  std::cout<< "transferring buffers...." << std::endl;
  return true;
}
