#include <flash_runtime/flashrt.h>
#include <ranges>
#include <algorithm>


std::shared_ptr<flash_rt> flash_rt::_global_ptr;


std::shared_ptr<flash_rt> EXPORT flash_rt::get_runtime( std::string runtime_lookup )
{

  if( _global_ptr )
    return _global_ptr;
  else
    return _global_ptr = std::shared_ptr<flash_rt>( new flash_rt( runtime_lookup) );

}

EXPORT flash_rt::flash_rt( std::string lookup)
: _backend( FlashableRuntimeFactory::Create( lookup ) )
{
  //get pointer to backend runtime
  _runtime_ptr = _backend.value()();
  std::cout << "Ctor'ing flash_rt...." << std::endl;

  std::cout << _backend->get_description() << std::endl; 
  
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

  auto sa = subaction{suba_id, num_of_inputs, rt_vars, kernel_args, exec_parms, opt};
  //add transactio and subactionsa
  _transactions.emplace( trans_id, sa );

  return {};
}

status EXPORT flash_rt::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
                                        std::string kernel_names[], std::optional<std::string> inputs[] ) 
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  std::vector<kernel_desc> kernel_inputs;

  auto pack_data = [&](int index)-> kernel_desc
  {
    return kernel_desc{kernel_types[index], kernel_names[index], inputs[index]};
  };

  //check if thier is a runtime existsa
  auto kernels = std::views::iota( (size_t) 0, num_kernels ) | std::views::transform(pack_data);
  
  //pack the inputs into
  std::ranges::for_each(kernels, [&](auto input){ kernel_inputs.push_back(input); } );
  
  if( _runtime_ptr )
  {
    _runtime_ptr->register_kernels( kernel_inputs  );  
  }else std::cout << "No runtime available" << std::endl;

 std::cout << "completed flash_rt::" << __func__ << std::endl;
 return {}; 
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

