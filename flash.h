#include <random>
#include <iostream>
#include <stdlib.h> 
#include <chrono>
#include <ranges>
#include <tuple>
#include <cstddef>
#include <algorithm>
#include <string>
#include <initializer_list>
#include <memory>
#include <optional>
#include <any>
#include <utils/common.h>
#include <vector>
#include <tuple>
#include <flash_runtime/flashrt.h>
#include <flash_runtime/flash_sync.h>
#include <flash_runtime/flash_part.h>
#include <flash_runtime/flash_memory.h>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////API relies on Runtime implementation thread-safety, and singleton//////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern size_t get_indices( int );

//RuntimeObj, SubmitObj, ExecObj, SnapShot
template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
struct SubmitObj;

template<typename _Upstream, typename _ExecParams, typename ... Ts> 
class RuntimeObj;

constexpr bool strings_equal(char const * a, char const * b) {
    return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

template<typename Upstream>
struct ExecObj
{
    ExecObj( Upstream upstream);

    void replay(){}

    private:
    
      std::optional<Upstream> _upstream;

};

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
struct SubmitObj
{
  using UpstreamUpstream_t         = Upstream::Upstream_t;
  using UpstreamRegistry_t         = Upstream::Registry_t;
  static constexpr size_t ParmSize = std::tuple_size_v<Parms>;

  SubmitObj(Upstream, Kernel, Parms, ParmPtrs);

  template<std::unsigned_integral ... Us>
  SubmitObj<Upstream, Kernel, Parms, ParmPtrs>& sizes(Us... sizes);

  template<std::unsigned_integral ... Us>
  ExecObj< SubmitObj<Upstream, Kernel, Parms, ParmPtrs > > exec(Us... items);

  template<std::unsigned_integral ... Us >
  auto defer(Us... items);

  template<typename ... Args>
  void reconcile_args( Args&& ... );

  private : 

    template<typename _Upstream, typename _ExecParams, typename ... Us>
    friend class RuntimeObj;

    auto operator =(auto rhs)
    {
      return rhs;
    }

    template<typename T>
    size_t _get_size( T&& arg );

    template<typename ... ParmAttrs>
    void _set_buffer_attrs( ParmAttrs... ); 

    void _set_directionality();

    template<typename ... Args>
    void _calc_sz_from_contrs(Args&& ...  );

    template<size_t N, typename Indices = std::make_index_sequence<N> >
    void _update_parm_ptrs();

    auto _get_buffer_ptrs() -> std::array<std::add_pointer_t<void>, ParmSize >;

     //start forward execution path
    template<typename ExecParams, typename ... Us>
    void _forward_exec(ulong, ulong&,  ExecParams Params,  Us ... );

    template<typename T> friend class ExecObj;

    Kernel _override_kernel;

    std::function<void()> _snapshot;
    std::optional<Upstream> _upstream;
    ParmPtrs _buffers;
    Parms    _shadow_buffers;
    std::array<size_t, ParmSize > _sizes;
    std::array<ParmAttr, ParmSize > _bufferAttrs;
};


template<typename _Upstream = NullType, typename _ExecParams = NullType, typename ... Ts>
class RuntimeObj 
{
  public :
    using Upstream_t    = _Upstream;
    using Registry_t    = std::tuple<Ts...>;
    using RuntimeImpl_t = std::shared_ptr<flash_rt>;
    using ExecParams_t  = _ExecParams;


    //Thei ctor will allow host code to utilize any implementation avaiable for
    //a specific kernel
    RuntimeObj()
    : _runtimeImpl(flash_rt::get_runtime("ALL") ) {}
    

    //constructor without upstream
    explicit  RuntimeObj(std::string rt_key, Ts&& ... ts)
    : _runtimeImpl( flash_rt::get_runtime(rt_key) ), _registry( ts...)
    {
      constexpr size_t N = sizeof...(ts);
      std::cout << "Calling RuntimeObj without Upstream..." << std::endl;
      //register all functions
      std::cout << std::boolalpha;
      std::string base_knames[N]  = { ts.get_method()... };
      kernel_t kernel_types[N]    = { ts.get_kernel_type()... };

      constexpr std::array<bool, N> is_same( {std::is_same_v<std::string, typename Ts::program_t>...} );
      constexpr bool all_type = std::all_of( is_same.begin(), is_same.end(), [](bool b){ return b; } );
  
      if constexpr( all_type )
      {
        std::cout << "All types are strings" << std::endl;
        std::optional<std::string> inputs[N] = { ts.get_kernel_details()... };

        _runtimeImpl->register_kernels(N, kernel_types, base_knames, nullptr, inputs);
      }
      else std::cout << "Different types detected" << std::endl;
      std::cout << "RuntimeObj Initialized" << std::endl;
    }

    auto operator =(auto rhs)
    {
      return rhs;
    }

    template<size_t I>
    auto get_ctor_input()
    {
       return std::get<I>(_registry).get_kernel_details(); 
    }

    Registry_t get_kernel_definition_registry()
    {
      return _registry;
    }

    template<size_t I>
    auto get_kernel_definition()
    {
      std::string name = std::get<I>(_registry).get_kernel_details().value();
      return std::get<I>(_registry);
    }

    template<typename T>
    constexpr auto _scalar_parm_conv(T&& arg)
    {
      if constexpr ( IsPointer<T> || std::is_rvalue_reference_v<decltype(arg)> )
      {
        return arg;
      }
      else /* scalar */
      { 
        return &arg;       
      }

    }
   
    template< typename KD, typename... Args >
    auto submit(KD kernel_def, Args&&... args)
    {  
        /* alot of implicit stuff warning */
        //the _scalar_parm_conv uses ra_Value check to create the 
        //shadow regideter types, 
        //all l-values are converterd to pointers 
        auto sv = SubmitObj(*this, kernel_def, 
                            std::make_tuple( _scalar_parm_conv(std::forward<Args>(args))... ),
                            std::make_tuple( _scalar_parm_conv(args)... ) );
        
        sv.reconcile_args( std::forward<Args>(args)... );
                
        return sv;
    }
 
    template<typename T>
    RuntimeObj<Upstream_t, ExecParams_t,  Ts...>& device_select(T devices) &&
    {
    //this is 
      //this is updating the temp object
      return *this;
    }

    template<typename T>
    RuntimeObj<Upstream_t, ExecParams_t, Ts...> device_select(T devices)
    {
      //copy ctor the current RuntimeObj
      auto RTObj = *this;

      return std::move( RTObj ).device_select( devices);
    }


    private: 

        template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
        friend struct SubmitObj;

        //constructor with upstream
        RuntimeObj(std::string rt_key, Upstream_t& upst, ExecParams_t exec_params, Ts&& ... ts)
        : _upstream(upst), _runtimeImpl(flash_rt::get_runtime(rt_key) ), _exec_params(exec_params), _registry( ts...)
        {
          std::cout << "Calling RuntimeObj with Upstream..." << std::endl;
         
        }

        RuntimeImpl_t get_runtime(){ return _runtimeImpl; }

        //This is incase a deferment occurs
        //std::optional<Upstream> _upstream;
        std::optional<Upstream_t>   _upstream;
        RuntimeImpl_t               _runtimeImpl;
        Registry_t                  _registry;
	std::optional<ExecParams_t> _exec_params;

        template< size_t N, typename Kernel>
        auto _get_directions( );
        
        template< typename P, typename PA, typename I, typename Kernel, typename ExecParams>
        void execute(auto trans_sub_id, Kernel kernel, P buffers, PA buffer_attrs,
                     auto sizes, ExecParams successor_params, I exec_items )
        {
            std::vector<size_t> exec_params;

            size_t num_inputs = Kernel::Get_NInArgs() + 
                                Kernel::Get_NInOutArgs();
            //std::cout << "Executing " << kernel_id << "..." << std::endl;
            constexpr size_t NArgs = std::tuple_size_v<P>;
           
            auto _te_buffers = erase_tuple( buffers, buffer_attrs, sizes );            

            auto _rt_vars    = runtime_vars{ kernel.get_method(),           //Base kernel name
                                             kernel.get_kernel_type(),
                                             kernel.get_kernel_details(),   //method_override
                                             kernel.get_method_ovr(),       //new method detials
                                             kernel.simplify_kattrs()  };

	    //set transactio information
            _rt_vars.associate_transactions( trans_sub_id );


	    if constexpr ( std::is_same_v<ExecParams, NullType>  )
	    {
              std::cout << "Exec : ";
              auto arr = get_array_from_tuple( exec_items );
	      exec_params  = std::vector<size_t>( arr.begin(), arr.end() );
            }
            else
            {
              std::cout << "Defer : ";
              auto arr = get_array_from_tuple( successor_params );
	      exec_params  = std::vector<size_t>( arr.begin(), arr.end() );
            }

            
	    std::ranges::copy(exec_params, std::ostream_iterator<size_t>{std::cout, ", "} );
	    std::cout << std::endl;

            _runtimeImpl->execute( _rt_vars, num_inputs, _te_buffers, exec_params, options{}); 
            //money maker: this function will interface with the runime Object
        }

        //does nothing in runtime obj
        template<typename ExecParams, typename ... Us>
        void _forward_exec( ulong tid, ulong& sa_id, ExecParams, Us ... items) 
        {
            if constexpr (std::is_same_v<NullType, _Upstream>) 
            {
              return;
            }
            else
            {
              if( _upstream ) _upstream->_forward_exec(tid, sa_id,
                                                       _exec_params.value(), items...);
              else std::cout << "Could not find upstream" << std::endl;    
            }
            
        }
    
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename _Upstream, typename _ExecParams, typename ... Ts>
template<size_t N, typename Kernel>
auto RuntimeObj<_Upstream, _ExecParams, Ts...>::_get_directions( )
{
  const size_t NInArgs    = Kernel::Get_NInArgs();
  const size_t NInOutArgs = Kernel::Get_NInOutArgs();
  const size_t NArgs      = NInArgs + NInOutArgs;

  std::array<DIRECTION, N> out;
  auto out_it = out.begin();
 
  static_assert((N < NArgs), "Not enough Parameters to submit kernel" );

  std::ranges::fill_n(std::next(out_it, 0),       NInArgs,    DIRECTION::IN     );
  std::ranges::fill_n(std::next(out_it, NInArgs), NInOutArgs, DIRECTION::INOUT  );
  std::ranges::fill_n(std::next(out_it, NArgs),   N-NArgs,    DIRECTION::IN     );

  return out;
}

//////////////////////////////////////////////////////////////Definitions///////////////////////////////////////////////////////////////////////////////////
template<typename Upstream>
ExecObj<Upstream>::ExecObj( Upstream upstream)
{
  _upstream = upstream;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<size_t N, typename Indices >
void SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::_update_parm_ptrs()
{
  auto move = [&]<typename T, typename U>( T&& rhs, U&& lhs ) 
  {
    if constexpr ( IsPointer<U> )
      return lhs;
    else
      return &lhs; 
  }; 

  auto start = [&]<size_t... I >(std::index_sequence<I...> )
  {
     bool vec[N] = { ( move( std::get<I>(_buffers), 
                             std::get<I>(_shadow_buffers) ), true)... };
  };

  start( Indices{} );
  
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::SubmitObj(Upstream upst, 
                                                        Kernel dynamic_override, 
                                                        Parms parms, ParmPtrs parm_ptrs)
{
    _upstream = upst;

    _override_kernel = dynamic_override;

    //root value holders
    _shadow_buffers = parms;
    //pointer holders
    _buffers = parm_ptrs;

    _update_parm_ptrs< std::tuple_size_v<ParmPtrs> >( );

    _sizes.fill(1);
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<std::unsigned_integral ... Us>
ExecObj< SubmitObj<Upstream, Kernel, Parms, ParmPtrs> > 
SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::exec(Us... items)
{
  std::cout << "Start exec..." << std::endl;
  ulong subaction_id = 0;

  //create a unique_id and makes sure thier is no conflicting Id
  ulong transaction_id = flash_rt::get_runtime()->create_transaction();
  //execute kernels from root node, forward
  _forward_exec(transaction_id, subaction_id, NullType{}, items...);

  flash_rt::get_runtime()->process_transaction( transaction_id );

  return ExecObj(*this);
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<std::unsigned_integral ... Us>
auto SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::defer(Us... items)
{
  std::cout << "Defering ... " << std::endl;
  auto func1 = [&]<std::size_t N=std::tuple_size_v<typename Upstream::Registry_t>, typename Indices = std::make_index_sequence<N>>()
  {
      auto func2 = [&]<std::size_t... I >(std::index_sequence<I...> ) 
      {
	auto exec_parms = std::make_tuple(items...);

        return RuntimeObj(_upstream->get_runtime(), *this, exec_parms, _upstream->template get_kernel_definition<I>() ... );
      };
    
    return func2(Indices{}); 
  };
   
  return func1();
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<std::unsigned_integral ... Us>
SubmitObj<Upstream, Kernel, Parms, ParmPtrs>& SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::sizes(Us... sizes)
{
  _sizes = { sizes... };

  return *this;
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template< typename ExecParams, typename ... Us>
void SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::_forward_exec(ulong trans_id, ulong& subaction_id, ExecParams params, Us ... items )
{  
  //will call prior forward excute until it gets to the root runtime object
  std::optional<NullType> OptNull;
  _upstream->_forward_exec(trans_id, subaction_id, OptNull, items...);

  //Executing 
  auto trans_sub_id = std::make_pair(trans_id, subaction_id );
  _upstream->execute(trans_sub_id, _override_kernel, _buffers, _bufferAttrs, _sizes, params, std::make_tuple(items...) );

  std::cout << "subaction id = " << subaction_id << std::endl;
  subaction_id++;
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<typename T>
size_t SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::_get_size( T&& arg )
{
  if constexpr ( IsContainer<T> || IsAttribute<T> )
  {
    std::cout << __func__ << " : size : " << arg.size() << std::endl;
    return arg.size();
  } 
  else
    return 1;
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<typename ... ParmAttrs>
void SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::_set_buffer_attrs( ParmAttrs... parm_attrs)
{
  _bufferAttrs = { ParmAttr{parm_attrs}... }; 
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
void  SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::_set_directionality()
{
  constexpr  size_t N          = ParmSize;
  constexpr  size_t NInArgs    = Kernel::Get_NInArgs();
  constexpr  size_t NInOutArgs = Kernel::Get_NInOutArgs();
  constexpr  size_t NArgs      = NInArgs + NInOutArgs;

  auto out_it = _bufferAttrs.begin();
 
  static_assert((N >= NArgs), "Not enough Parameters to submit kernel" );

  using namespace std::ranges; 
  for_each_n(std::next(out_it, 0),       NInArgs,    [](auto& attr) { attr.dir = DIRECTION::IN;    } );
  for_each_n(std::next(out_it, NInArgs), NInOutArgs, [](auto& attr) { attr.dir = DIRECTION::INOUT; } );
  for_each_n(std::next(out_it, NArgs),   NInArgs,    [](auto& attr) { attr.dir = DIRECTION::OUT;   } );
}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<typename ... Args>
void SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::reconcile_args( Args&&... args)
{
 
  std::cout << __func__ << " : Mark 1" << std::endl;
  std::vector<ParmAttr > arg_type = 
  { ParmAttr{ IsPointer<decltype(args)>, 
              IsContainer<decltype(args)>, 
              IsAttribute<decltype(args)>, 
              std::is_rvalue_reference_v<decltype(args)>  } ... 
  };
  
  for( auto parm : arg_type ) 
    std::cout << "IsContainer : " << parm.is_container << std::endl;	

  std::cout << __func__ << " : Mark 2" << std::endl;
  //set default sizes to container size
  sizes( _get_size(args)... );
  // set variable attributes
  std::cout << __func__ << " : Mark 3" << std::endl;
  int i = 0;
  _set_buffer_attrs( (args, arg_type.at(i++) )... );
  //reconcile scalar variables
  std::cout << __func__ << " : Mark 4" << std::endl;
  _set_directionality( );

}

template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
auto SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::_get_buffer_ptrs() ->
std::array<std::add_pointer_t<void>, ParmSize >
{
  std::array<void*, ParmSize > out;

  auto contr_move = [&]<typename T>( T arg ) 
  { 
    if constexpr( IsContainer< std::remove_pointer_t<T> > )
    { 
      std::cout << __func__ << " : found container " << std::endl;
      return arg->data();
    } else return arg;
  };

  auto move = [&]<size_t... I >(std::index_sequence<I...> )
  { 
    
    out = { reinterpret_cast<void *>(contr_move(std::get<I>(_buffers)) )... };  
  };

  move( std::make_index_sequence<ParmSize>{} );
  return out;  
}


template<typename Upstream, typename Kernel, typename Parms, typename ParmPtrs>
template<typename ... Args>
void SubmitObj<Upstream, Kernel, Parms, ParmPtrs>::_calc_sz_from_contrs(Args&&... args)
{
  std::cout << __func__ << " : size " <<  _sizes.size() <<  std::endl; 
  size_t arg_cnt = sizeof...(args);
  auto arg_idx   = std::views::iota(( size_t)0, arg_cnt);
  auto ptr_list = _get_buffer_ptrs( );

  auto is_container = [&](auto attr_idx) { 
                          auto b =  _bufferAttrs.at(attr_idx).is_container;  
                          return b;
                          };
  //calculate sizes based on containers
  //Go through each contrainer variable
  for( auto contr_idx : arg_idx | std::views::filter( is_container ) )
  {
    std::cout << __func__ << " : container index : " << contr_idx << std::endl; 
    auto contr_size  = _sizes.at(contr_idx);
    auto contr_ptr   =  ptr_list[contr_idx];
    auto contr_tsize = _bufferAttrs.at(contr_idx).type_size;
    auto non_contr   = arg_idx | std::views::filter( unary_diff{contr_idx}) | 
                       std::views::filter( [&](auto i){ return !_bufferAttrs.at(i).is_container; } );
    std::vector<std::pair<size_t,size_t > > overlap;
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //then go through each non-container_variable and reset the sizes
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    for(auto idx : non_contr ) 
    {              
      auto cand_ptr  = ptr_list[idx];
      ptrdiff_t delta = ( (ptrdiff_t) cand_ptr - 
                          (ptrdiff_t) contr_ptr ) / contr_tsize; 
      
      if( (contr_ptr < cand_ptr) && (delta < contr_size)  )
      {
        _sizes.at(idx) = contr_size - delta;
        overlap.push_back( {idx, _sizes.at(idx) } );
        std::cout << __func__ << " : container size : " << contr_size << " : " 
                  << idx << " : " << _sizes.at(idx) << std::endl; 
      }
          
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    std::ranges::sort( overlap, {}, &decltype(overlap)::value_type::second ); 
    
  } 
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<std::size_t N>
struct KernelDECL
{
    char p[N]{};

    constexpr KernelDECL ( char const(&pp)[N] )
    {
        std::ranges::copy(pp, p);
    };

};

template<size_t NumPureInputs, KernelDECL KD, kernel_t k_type=kernel_t::INT_SRC, typename ... Ts>
struct KernelDefinition
{
    using input_ts = std::tuple<Ts...>;
    //using program_t = kernel_t_decl<k_type>;
    using program_t = typename kernel_t_decl<k_type>::value_type;
    
    constexpr std::string get_method()        { return std::string(KD.p); }
    constexpr kernel_t    get_kernel_type()   { return k_type; }
    kernel_t_decl<k_type> get_kernel_details(){ return _input_program; }
    auto                  get_method_ovr()    { return _kernel_name; }

    KernelDefinition ( kernel_t_decl<k_type> inp={}, 
                       std::optional<std::string> kernel_name={} )
    : _input_program(inp), _kernel_name(kernel_name) { }


    static constexpr size_t Get_NArgs()
    {
      return sizeof...(Ts);
    }

    static constexpr size_t Get_NInArgs() { 
      return NumPureInputs;
    }

    static constexpr size_t Get_NAttrs() {
      return std::tuple_size_v<decltype(_attrs)>;
    }

    static constexpr size_t Get_NInOutArgs() {
      return Get_NArgs() - (Get_NInArgs() + Get_NAttrs() );
    }

    std::vector<te_attr> simplify_kattrs()
    {
      constexpr size_t n_attr = std::tuple_size_v<decltype(_attrs)>;

      auto conv = [&]<size_t... I>(std::index_sequence<I...> )
      {
         std::vector<te_attr> out = { te_attr
                                      {
                                       std::get<I>(_attrs).get_id(),
                                       std::get<I>(_attrs).get_gen_vals()
                                      }...
                                    }; 
         return out;
      };

      return conv( std::make_index_sequence<n_attr>{} );

    }

    private:

      template<size_t N, typename T, typename ... Us>
      struct sub_type_list
      {
        static constexpr int accumulate( )
        {
          if constexpr( IsAttribute<T> ) return sub_type_list<K, Us...>::accumulate( );
          else return N;
        }

        static constexpr int K = N + 1;
      };

      template<typename... Us>
      struct attr_list
      { 
        using args = std::tuple<Us...>;

        static constexpr int attr_n = sub_type_list<0, Us...>::accumulate();
           
        static constexpr auto create_default_attr_tuple()
        {
          auto calc = [&]<size_t... I >(std::index_sequence<I...> )
          {
            return std::tuple<std::tuple_element_t<I, args>...>{};
          };

          auto  ret = calc( std::make_index_sequence<attr_n>{} );
          return ret;
        }
        
        static constexpr auto create()
        { 
          return create_default_attr_tuple();
        }
 
      };


      decltype(auto) get_attrs() 
      {
        return (_attrs); 
      }


      decltype( attr_list<Ts...>::create() ) _attrs;
      //static auto _attrs  = attr_list<Ts...>::create();
      kernel_t_decl<k_type> _input_program;
      std::optional<std::string> _kernel_name;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////a
