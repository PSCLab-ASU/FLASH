#include <flash_runtime/flashrt.h>
#include <ranges>
#include <algorithm>


std::shared_ptr<flash_rt> flash_rt::_global_ptr;


std::shared_ptr<flash_rt> flash_rt::get_runtime( std::string runtime_lookup )
{

  if( _global_ptr )
    return _global_ptr;
  else
    return _global_ptr = std::shared_ptr<flash_rt>( new flash_rt( runtime_lookup) );

}

flash_rt::flash_rt( std::string lookup)
: _backend( FlashableRuntimeFactory::Create( lookup ) )
{
  //get pointer to backend runtime
  _runtime_ptr = _backend.value()();
  std::cout << "Ctor'ing flash_rt...." << std::endl;

  std::cout << _backend->get_description() << std::endl; 
  
}

status flash_rt::execute(runtime_vars rt_vars,  uint num_of_inputs, 
                         std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms, options opt)
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  //need to store all arguments for later processing through process_transaction
  auto[trans_id, suba_id] = rt_vars.get_ids();
  //create a subactions
  auto sa = subaction{suba_id, num_of_inputs, rt_vars, kernel_args, exec_parms, opt};
  //add transactio and subactionsa
  _transactions.emplace( trans_id, sa );

  return {};
}

status flash_rt::register_kernels( size_t num_kernels, kernel_t kernel_types[], 
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

ulong flash_rt::create_transaction()
{
  ulong tid = random_number();

  return tid;
}

status flash_rt::process_transaction( ulong tid )
{
  std::cout << "calling flash_rt::" << __func__ << std::endl;
  ///////////////////////////////////////////////////////////
  std::vector<status> statuses;
  auto [b_iter, e_iter] = _transactions.equal_range(tid);

  
  if( !_runtime_ptr ) std::runtime_error("No backend selected/available!");

  if( auto dist = std::distance(b_iter, e_iter); dist != 0  )
  {
    //check if thier is a runtime exists
    //std::vector<std::reference_wrapper<subaction> > pipeline(b_iter, e_iter);
    std::vector<std::reference_wrapper<subaction> > pipeline;

    std::transform( b_iter, e_iter, std::back_inserter( pipeline ), []( auto subacts )
    {
      return std::ref(subacts.second); 
    } );

    //sort by subaction id
    std::ranges::sort( pipeline, {}, &subaction::subaction_id);

    //submit subactions
    std::for_each( pipeline.begin(), pipeline.end(), [&](subaction& stage)
    {
      auto[num_of_inputs, rt_vars, kernel_args, exec_parms] = stage.input_vars();
      statuses.push_back( _runtime_ptr->execute(rt_vars, num_of_inputs, kernel_args, exec_parms ) );

    });

    ///wait for work to complete
    auto completed = statuses | std::views::filter( unary_equals{true} ) |  std::views::transform([&](auto stat)
    {
      if( stat.work_id )
      {
        auto wid = stat.work_id.value();
        return _runtime_ptr->wait( wid );
      }
      else 
      {
        std::cout << "Could not find wid" << std::endl;
        return status{-1};
      }

    } ); 

    auto all_complete = std::ranges::all_of( completed, unary_equals{0}, &status::err );
    if( all_complete ) std::cout << "successfully completed tid = " << tid << std::endl;
    else std::cout << "failed to complete tid = " << tid << std::endl; 

  }
  else
  {
    std::cout << "Could not locate transaction " << tid << std::endl;
  }

  return {};
}

