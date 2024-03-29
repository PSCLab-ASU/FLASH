#include <flash_runtime/flashrt.h>
#include <ranges>
#include <algorithm>


//std::shared_ptr<flash_rt> flash_rt::_global_ptr;

//use for kernels function
extern size_t EXPORT get_indices( int ind);


std::shared_ptr<flash_rt> EXPORT flash_rt::get_runtime( std::string runtime_lookup )
{

  if( _global_ptr )
  {
    return _global_ptr->_customize_runtime(runtime_lookup);
  }
  else
  {
    _global_ptr = std::shared_ptr<flash_rt>( new flash_rt( "ALL" ) );

    return _global_ptr->_customize_runtime(runtime_lookup);
  }
}

std::shared_ptr<flash_rt> flash_rt::_customize_runtime( std::string rt_key )
{
  return _rtrs_tracker.get_create_runtime( rt_key );
}


EXPORT flash_rt::flash_rt( std::string lookup)
{

  if( lookup != "ALL")
  {
    _runtime_key = lookup;
    _backend = FlashableRuntimeFactory::Create( lookup );
    //get pointer to backend runtime
    _runtime_ptr = _backend.value()();
    std::cout << "Ctor'ing flash_rt...." << std::endl;

    std::cout << _backend->get_description() << std::endl; 
  }
  else
  {
    //this indicates to the system that it should
    //choose the runtime
    _runtime_ptr = nullptr;
  }
}

std::string flash_rt::_recommend_runtime( const std::string& kernel_name,
                                          const std::vector<te_variable>& kernel_args )
{
  auto kernel_rts = _rtrs_tracker.get_all_runtimes_by( kernel_name );
  std::vector<std::string> kernel_rtks, buffer_rtks;

  std::ranges::transform( kernel_rts, std::back_inserter(kernel_rtks),
                          [](auto kernel_rt)
  {
    auto rt =  kernel_rt->get_runtime_key();
    std::cout << "Found runtime : " << rt.value_or(g_NoRuntime) << std::endl;
    return rt.value_or(g_NoRuntime);

  } );

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
    auto runtime = _rtrs_tracker.get_runtime_by_fmem( fmem_id );
    if( runtime )
      return *runtime->get_runtime_key();
    else
      return g_NoAlloc;

  } );
  ///////////////////////////////////////////////////////////////////////////////////////////// 
  auto none_alloc    = std::ranges::all_of( buffer_rtks, unary_equals{ std::string("NoAlloc") } );  
  auto is_colocated  = std::ranges::all_of( buffer_rtks, unary_equals{ kernel_rtks[0] } );  

  if( none_alloc ) 
    std::cout << "None of the flash_memory buffers allocated! : " << kernel_rtks[0] << std::endl;
  else if( is_colocated ) 
    std::cout << "Kernel and buffers are colocated in "<< kernel_rtks[0] << std::endl; 
  else
    std::cout << "Mix of runtimes found!" << std::endl; 

  return kernel_rtks[0];
}

void flash_rt::_manage_buffers( std::string tid, std::string rtk, std::vector<te_variable>& kernel_args )
{
  auto exec_rt = _rtrs_tracker.get_create_runtime( rtk );

  auto fmem_pred = [](auto arg) -> bool
  {  
    return arg.is_flash_mem();
  };

  auto fmems = kernel_args | std::views::filter(fmem_pred);
  
  for( auto& fmem : fmems )
  {
    //auto buffer_rt  = _rtrs_tracker.get_runtime_by_fmem( std::to_string(fmem.get_fmem_id()) );
    auto buffer_rt  = _rtrs_tracker.get_runtime_by_fmem( fmem.get_fmem_id() );
    auto buffer_rtk = buffer_rt?buffer_rt->get_runtime_key():g_NoAlloc;

    if( rtk == buffer_rtk ) 
      std::cout << "Buffer " << fmem.get_fmem_id() << " already allocated" << std::endl;
    else if ( buffer_rtk == g_NoAlloc )
    {
      bool res = exec_rt->allocate_buffer(fmem);
      if( res ) _rtrs_tracker.register_fmem( tid, rtk, fmem );
    }
    else if( rtk != buffer_rtk )
      //transaction id, src_rt, dst_rt, buffer_info
      _transfer_buffer(tid, buffer_rt, exec_rt, fmem ); 

  }

}

status flash_rt::_transfer_buffer( std::string, std::shared_ptr<flash_rt> buffer_rt, 
                                   std::shared_ptr<flash_rt> exec_rt, te_variable& fmem )
{


  return status{};
}
        


status EXPORT flash_rt::execute(runtime_vars rt_vars,  uint num_of_inputs, 
                         std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms, options opt)
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

  if( _runtime_key == "ALL" )
  {
    auto kname_ovr   = rt_vars.get_kname_ovr();
    auto kernel_name = kname_ovr.value_or( rt_vars.get_lookup() );

    rtk = _recommend_runtime( kernel_name, kernel_args);
  }
  
  _manage_buffers( std::to_string(trans_id), rtk, kernel_args );

  auto sa = subaction{suba_id, num_of_inputs, rt_vars, kernel_args, exec_parms, opt};
  //add transactio and subactionsa
  _transactions.emplace( trans_id, sa );

  return {};
}

status EXPORT flash_rt::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                          std::string kernel_names[], std::optional<std::string> kname_ovrs[],
                                          std::optional<std::string> inputs[] ) 
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;

  std::vector<kernel_desc> kernel_inputs;

  auto pack_data = [&](int index)-> kernel_desc
  {
    if( kname_ovrs == nullptr )
      return kernel_desc{kernel_types[index], kernel_names[index], 
                         kernel_names[index], inputs[index]};
    else
      return kernel_desc{kernel_types[index], kernel_names[index], 
                         kname_ovrs[index],   inputs[index]};
   
  };

  //check if thier is a runtime existsa
  auto kernels = std::views::iota( (size_t) 0, num_kernels ) | std::views::transform(pack_data);
  
  //pack the inputs into
  std::ranges::for_each(kernels, [&](auto input){ kernel_inputs.push_back(input); } );
  
  if( _runtime_ptr )
  {
    std::vector<bool> successes;
    _runtime_ptr->register_kernels( kernel_inputs, successes );

    //go through each success registration and add it to the registry
    std::ranges::transform(kernel_inputs, successes, std::back_inserter(successes),
    [&](kernel_desc& kd, const bool& success)
    {
      if( success )  _register_kernel( _runtime_key.value(), kd);
      return false;
    } );
       
  }
  else std::cout << "No runtime selected for registration" << std::endl;

 std::cout << "completed flash_rt::" << __func__ << std::endl;
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
  std::vector<status> statuses;
 
  auto [b_iter, e_iter] = _transactions.equal_range(tid);

  
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

    std::cout << "---- sorting complete" << std::endl;
    //submit subactions
    std::for_each( pipeline.begin(), pipeline.end(), [&](subaction& stage)
    {
      auto[num_of_inputs, rt_vars, kernel_args, exec_parms] = stage.input_vars();

      auto ktype = rt_vars.get_ktype();
      auto base_kname = rt_vars.get_lookup();
      auto kname_ovr  = rt_vars.get_kname_ovr();
      auto kname_impl = rt_vars.get_kimpl();
      register_kernels( 1, &ktype, &base_kname, &kname_ovr, &kname_impl ); 
                           
      statuses.push_back( _runtime_ptr->execute(rt_vars, num_of_inputs, kernel_args, exec_parms ) );

    });

    std::cout << "---- foreach complete" << std::endl;
    ///wait for work to complete
    auto completed = statuses | std::views::filter( unary_equals{true} ) |  std::views::transform([&](auto stat)
    {
      if( stat.work_id )
      {
        auto wid = stat.work_id.value();
        std::cout << "waiting on " << wid << " to complete..." << std::endl;
        return _runtime_ptr->wait( wid );
        std::cout << "completed " << wid << std::endl;
      }
      else 
      {
        std::cout << "Could not find wid" << std::endl;
        return status{-1};
      }

    } ); 

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

  return {};
}

void flash_rt::_register_kernel(std::string rtk, const kernel_desc& kd)
{
  _rtrs_tracker.register_kernel(rtk, kd );
}



/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
runtimes_resource_tracker::shared_flash_runtime
runtimes_resource_tracker::get_create_runtime(std::string rtk )
{
  if( _runtime_ptrs.count( rtk ) == 0 )
  {
    //auto srtp = std::make_shared<flash_rt>(rtk);
    auto srtp = std::shared_ptr<flash_rt>(new flash_rt( rtk) );
    _runtime_ptrs[rtk] = srtp;
  }

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
runtimes_resource_tracker::get_runtime_by_fmem( std::string fmem_id )
{
  auto rt_sptr = _get_runtime_by( fmem_id );
  return rt_sptr;
}

runtimes_resource_tracker::shared_flash_runtime
runtimes_resource_tracker::_get_runtime_by( std::string id )
{

  auto rts = get_all_runtimes_by( id );
 
  if( rts.size() == 0 ) {
   std::range_error("Could not find valid runtime for: " + id );
  }

  return rts[0];
}

void
runtimes_resource_tracker::register_fmem( std::string tid, std::string rtk, const te_variable& fmem )
{
  
  summary_flash_mem sfm;
  sfm.tid       = tid; 
  //sfm.id        = std::to_string( fmem.get_fmem_id() );
  sfm.id        = fmem.get_fmem_id();
  sfm.type_size = fmem.type_size;
  sfm.vec_size  = fmem.vec_size;
  sfm.base_addr = fmem.data;

  _resources.emplace( rtk, sfm );
  

}

auto 
runtimes_resource_tracker::get_all_runtimes_by( const std::string& id )
-> std::vector<shared_flash_runtime>
{
  std::vector<shared_flash_runtime> out;

  for(auto[rtk, summary_obj] : _resources )
  {
    std::visit([&](auto&& obj)
    {
      if constexpr (std::is_same_v<decltype(obj), summary_kernel > )
      {
        if( obj.kernel_name == id )
          out.push_back( get_create_runtime( rtk ) ); 
      }
      else if constexpr (std::is_same_v<decltype(obj), summary_flash_mem> )
      {
        if( obj.id == id )
          out.push_back( get_create_runtime( rtk ) ); 
      }
    }, summary_obj);
    //break if we found a runtime
  }

  return out;
}

void
runtimes_resource_tracker::register_kernel( std::string rtk, const kernel_desc& kd )
{
  auto base_kname = kd.get_base_kname();             //kernel name
  auto base_loc   = _get_base_loc(rtk, base_kname);  //location
  auto ovr_kname  = kd.get_ovr_kname();            //optional kernel name
  auto ovr_loc    = kd.get_kernel_def();             //optional location

  std::string kname = ovr_kname?ovr_kname.value():base_kname;
  std::string loc   = ovr_loc?ovr_loc.value():base_loc;

  summary_kernel sk{ kname, loc };

  _resources.emplace(rtk, sk);
}

std::string runtimes_resource_tracker::_get_base_loc( std::string rtk, std::string kname)
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


bool runtimes_resource_tracker::kernel_exists( std::string rtk, std::string kname, std::string kernel_impl)
{
  return true;
}

bool runtimes_resource_tracker::_transfer_buffer( std::string tid, shared_flash_runtime src, 
                                                  shared_flash_runtime dst, te_variable& fmem )
{
  //TBD dont foret to deallocate from src runtime.
  std::cout<< "transferring buffers...." << std::endl;
  return true;
}
