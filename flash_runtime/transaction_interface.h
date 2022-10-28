#include <utils/common.h>
#include <optional>
#include <variant>
#include <thread>
#include <condition_variable>


#pragma once


struct subaction
{
  ulong subaction_id;
  std::string runtime_key;
  uint num_inputs;
  runtime_vars rt_vars;
  std::vector<te_variable> kernel_args;
  std::vector<size_t> exec_parms;
  v_options lopts;
  std::function<int()> pre_pred, post_pred;

  //first = true, last = false, none = intermediate
  std::optional<bool> first_last_int; 
  bool done=false;

  //table references
  o_string _index_table_key;

  void set_itable_key( std::string key )
  {
    _index_table_key = key;
  }

  void set_first(){ first_last_int = true;  }
  bool is_first() { return first_last_int.value_or(false); }

  void set_last() { first_last_int = false; }
  bool is_last() { 
    if( first_last_int )
      return ( first_last_int.value() == false );
    else return false;
  }

  void set_intrm(){ first_last_int.reset(); }
  void mark_complete(){ done = true; }

  bool is_intrm() { return !( (bool) first_last_int); }
  bool is_done() { return done; }

  v_options  get_options(){ return lopts; }

  te_attrs get_kattrs(){ return get_rtvars().kAttrs; }

  ulong get_saId() { return subaction_id; }

  std::string get_rtk()
  {
    return runtime_key;
  }

  void set_preds( std::function<int()>&& pre, std::function<int()>&& post)
  {
    pre_pred  = pre;
    post_pred = post;
  }

  bool intersect( const subaction& ) const;

  const runtime_vars& get_rtvars(){ return rt_vars; }

  bool need_index_table() { return ( get_kattrs().size() > 0); }

  auto input_vars( ) 
  { 
    return std::make_tuple(std::ref(num_inputs), 
                           std::ref(rt_vars), 
                           std::ref(kernel_args), 
                           std::ref(exec_parms), 
                           std::ref(pre_pred), 
                           std::ref(post_pred) );
  };

};

class transaction_interface;

struct transaction_vars
{

  friend transaction_interface;
  using ids = std::pair<ulong, ulong>;
  using event_var = std::pair<std::mutex, 
                              std::condition_variable>;

  private:
    std::optional<ulong> _tid=0;
    std::multimap<ulong, subaction> _transactions;
    std::map<ulong, option> _gopts;
    std::map<ids, event_var> _events;

    /*~transaction_vars()
    {
      _events.clear();
      _gopts.clear();
      _transactions.clear();
    }*/
};

class flash_rt;

class transaction_interface : protected transaction_vars
{

  //flash rt is able to add transaction in
  friend flash_rt;

  //able to read existing transaction data interface
  public:
  
    transaction_interface& operator()( ulong tid)
    {
      _tid = tid;
      return *this;
    }

    ulong get_current_tid() const
    {
      return _tid.value_or(0);
    }
 
    template<typename T>
    bool check_option( ulong sa_id, T ops, std::optional<ulong> tid_ovr = {} )
    {
      bool exists = false;
      auto tr_id = tid_ovr.value_or( _tid.value() );
      auto& sa_payload = find_sa_within_ta(sa_id, tr_id);
      auto& sa_options = sa_payload.lopts;
      auto  rt_options = sa_payload.rt_vars.get_rtops();

      auto check_opt = std::bind( &option::check<T>, std::placeholders::_1, ops );
  
      //Checks to see if subaction option exists
      exists = std::ranges::any_of( sa_options, check_opt );
      //Checks to see if runtime options exists
      exists |= std::ranges::any_of( rt_options, check_opt );

      return exists;
      
    }

    v_options get_options( ulong, std::optional<ulong> tid = {} );

    std::pair<std::function<int()>, std::function<int()> >
    get_pred( ulong sa_id );

    subaction& find_sa_within_ta( ulong, std::optional<ulong> = {} );

    event_var& get_event( ulong, std::optional<ulong> = {} );

    void demarc_boundaries( std::optional<ulong> = {} );

  //transaction interface can only be instantiated by the flash_rt 
  protected:
  
    transaction_interface( );
 
    subaction& add_sa2ta(ulong, subaction&& );

    auto get_transaction( ulong tid )
    {
      auto exists = _transactions.contains(tid); 
      printf("_transactions.contain() : %i \n", exists );
      if ( !exists )
	throw std::runtime_error("Could not find transaction...");
  
      return _transactions.equal_range(tid);

    }


};


