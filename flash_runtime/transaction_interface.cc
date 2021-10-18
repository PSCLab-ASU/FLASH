#include <utils/common.h>
#include <flash_runtime/flashrt.h>


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////subaction class/////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
bool subaction::intersect(const subaction& dep_sa) const
{
  for( auto dep_karg : dep_sa.kernel_args )
  {
    auto dep_mem_id = dep_karg.get_mem_id();

    for(auto tar_karg : kernel_args )
    {
      auto tar_mem_id = tar_karg.get_mem_id();

      if( (dep_karg.parm_attr.dir != DIRECTION::IN) &&
           dep_mem_id == tar_mem_id )
        return true;
    }
  }

  return false;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////transaction_interface///////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

transaction_interface::transaction_interface()
{

}


subaction&
transaction_interface::add_sa2ta( ulong trans_id, subaction&& sa)
{
  return _transactions.emplace( trans_id, sa )->second;
}


template<typename T>
bool transaction_interface::check_option( ulong sa_id, T option, std::optional<ulong> tid_ovr)
{
  bool ret = false;
  auto tr_id = tid_ovr.value_or( _tid.value() );
  auto& sa_payload = find_sa_within_ta(sa_id, tr_id);
  auto& options = sa_payload.lopts;

  //run through all the options
  std::ranges::for_each( options, [&](auto& opt)
  {
     std::visit([&](auto& opt_var)
     {
       if( ( (short) opt_var) == ( (short) option) )
         ret = true; 

     }, opt.opt );

  } );

  return ret;

}

options transaction_interface::get_options(ulong sa_id, std::optional<ulong> tid_ovr ) 
{
  bool ret = false;
  auto tr_id = tid_ovr.value_or( _tid.value() );

  auto& sa_payload = find_sa_within_ta(sa_id, tr_id);

  return sa_payload.lopts; 
} 

transaction_vars::event_var&
transaction_interface::get_event( ulong sa_id, std::optional<ulong> tid_ovr )
{
  ulong tr_id = tid_ovr.value_or( _tid.value() );

  std::pair<ulong,ulong> key = { tr_id, sa_id };

  return _events.at(key);
}

void 
transaction_interface::demarc_boundaries( std::optional<ulong> tid_ovr)
{
  auto tr_id = tid_ovr.value_or( _tid.value() );
  auto cnt   = _transactions.count( tr_id );  

  auto start_sa_It = _transactions.lower_bound( tr_id );

  start_sa_It->second.set_first();  
  std::next(start_sa_It, cnt - 1)->second.set_last();
}


std::pair<std::function<int()>, std::function<int()> >
transaction_interface::get_pred( ulong sa_id )
{
  std::function<int()> dep_pred, succ_pred;
  std::vector<ids> dep_event_ids;

  auto [start_sa_It, end_sa_It] = _transactions.equal_range( _tid.value() );

  auto target_sa = find_sa_within_ta( sa_id );

  ids target_id = std::make_pair(_tid.value(), sa_id);
  auto& [t_mu, t_cv] = _events.at( target_id );

  //lock current subaction
  t_mu.lock();
  ///////////////////////////////////////////////////////////////////////////
  std::for_each(start_sa_It, end_sa_It, [&](auto& subas)
  {
    auto& [trans_id, sa] = subas;
    std::pair<ulong, ulong> id_dep = { trans_id, sa.get_saId() };   

    bool dep = target_sa.intersect( sa );
    bool itself = target_sa.get_saId() == sa.get_saId();

    if( dep && !itself ) 
      dep_event_ids.emplace_back( id_dep );
   
  } );
  ///////////////////////////////////////////////////////////////////////////
  //check if all the pre subaction are complete
  auto _func = [this, dep_event_ids, target_id](bool pre)
  {
    return [this, dep_event_ids, pre, target_id]()->int
    {
      for(auto[tid, sa_id] : dep_event_ids ) 
      {
        auto& [mu, cv] = this->get_event( tid, sa_id );
        if( pre )
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk);
        } 
        else cv.notify_all();
        
      }

      //unlock target on the succession
      if( !pre ) 
      { 
        auto& [t_mu, t_cv] = this->get_event( target_id.first, 
                                              target_id.second );
        t_mu.unlock();
      }

      return 0;
    };

  };
  //////////////////////////////////////////////////////////////////////////
  return std::make_pair( _func(true), _func(false) );
}

subaction& transaction_interface::find_sa_within_ta(ulong sa_id, std::optional<ulong> trans_id )
{
  ulong tid = trans_id.value_or( _tid.value() );

  auto [start_sa_It, end_sa_It] = _transactions.equal_range( tid );

  auto sa = std::ranges::find_if(start_sa_It, end_sa_It, [&](auto& sa)
            {
              return sa.second.get_saId() == sa_id;
            });

  return sa->second;

}

template<>
bool transaction_interface::check_option( ulong, trans_options, std::optional<ulong> );
