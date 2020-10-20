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
#include <common.h>
#include <vector>
#include <flash_runtime/flashrt.h>
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////API relies on Runtime implementation thread-safety, and singleton//////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct NullType {};

//RuntimeObj, SubmitObj, ExecObj, SnapShot
template<typename Upstream, typename NumInputs, typename Kernel, typename ... Ts>
struct SubmitObj;

template<typename _Upstream, typename RuntimeImpl, typename ... Ts> 
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

template<typename Upstream, typename NumInputs, typename Kernel, typename ... Ts>
struct SubmitObj
{
  using UpstreamUpstream_t = Upstream::Upstream_t;
  using UpstreamRegistry_t = Upstream::Registry_t;

  SubmitObj(Upstream upst, NumInputs, Kernel dynamic_override, std::function<void()> snapshot, Ts... ts);

  template<std::unsigned_integral ... Us>
  SubmitObj<Upstream, NumInputs, Kernel, Ts...>& sizes(Us... sizes);

  template<std::unsigned_integral ... Us>
  ExecObj< SubmitObj<Upstream, NumInputs, Kernel, Ts...> > exec(Us... items);

  template<std::unsigned_integral ... Us >
  auto defer(Us... items);


  private : 

    template<typename RuntimeImpl, typename _Upstream, typename ... Us>
    friend class RuntimeObj;

    //start forward execution path
    template<typename ... Us>
    void _forward_exec(ulong, ulong&, Us ... );

    template<typename T> friend class ExecObj;

    Kernel _override_kernel;
    constexpr static size_t _num_inputs=NumInputs::value, 
                            _num_outputs=sizeof...(Ts) - NumInputs::value;

    std::function<void()> _snapshot;
    std::optional<Upstream> _upstream;
    std::array<size_t, sizeof...(Ts)> _sizes;
    std::tuple<Ts...> _buffers;
};


template<typename RuntimeImpl, typename _Upstream = NullType, typename ... Ts>
class RuntimeObj 
{
  public :
    using Upstream_t    = _Upstream;
    using Registry_t    = std::tuple<Ts...>;
    using RuntimeImpl_t = RuntimeImpl;

    //constructor without upstream
    explicit  RuntimeObj(RuntimeImpl impl, Ts&& ... ts)
    : _runtimeImpl(impl), _registry( ts...)
    {
      constexpr size_t N = sizeof...(ts);
      std::cout << "Calling RuntimeObj without Upstream..." << std::endl;
      //register all functions
      std::cout << std::boolalpha;
      std::string keys[N]   = { ts.get_method()... };
      kernel_t kernel_types[N] = { ts.get_kernel_type()... };

      constexpr std::array<bool, N> is_same( {std::is_same_v<std::string, typename Ts::program_t>...} );
      constexpr bool all_type = std::all_of( is_same.begin(), is_same.end(), [](bool b){ return b; } );
  
      if constexpr( all_type )
      {
        std::cout << "All types are strings" << std::endl;
        std::optional<std::string> inputs[N] = { ts.get_kernel_details()... };

        _runtimeImpl->register_kernels(N, kernel_types,  keys, inputs);
      }
      else std::cout << "Different types detected" << std::endl;
      std::cout << "RuntimeObj Initialized" << std::endl;
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

    template< typename InpTL, typename ... Inputs, typename... Outputs, 
          std::size_t N=std::tuple_size_v<typename InpTL::input_ts>, typename Indices = std::make_index_sequence<N>>
    auto submit(InpTL lookup, Inputs... ins, Outputs... outs)
    {
        
        //TBD differentiate
        return SubmitObj(*this, std::integral_constant<size_t, N>{},  lookup, snap_shot(), outs... );
    }
 
    template<typename T>
    RuntimeObj<_Upstream, RuntimeImpl,Ts...>& device_select(T devices) &&
    {
    //this is 
      //this is updating the temp object
      return *this;
    }

    template<typename T>
    RuntimeObj<_Upstream, RuntimeImpl,Ts...> device_select(T devices)
    {
      //copy ctor the current RuntimeObj
      RuntimeObj<RuntimeImpl,Ts...> RTObj = *this;

      return std::move( RTObj ).device_select( devices);
    }


    private: 

        template<typename Upstream, typename NumInputs, typename Kernel, typename ... Us>
        friend struct SubmitObj;

        //constructor with upstream
        RuntimeObj(RuntimeImpl impl, _Upstream& upst, Ts&& ... ts)
        : _upstream(upst), _runtimeImpl(impl), _registry( ts...)
        {
          std::cout << "Calling RuntimeObj with Upstream..." << std::endl;
         

 
        }

        RuntimeImpl get_runtime(){ return _runtimeImpl; }

        //This is incase a deferment occurs
        //std::optional<Upstream> _upstream;
        std::optional<_Upstream> _upstream;
        RuntimeImpl _runtimeImpl;
        Registry_t _registry;

        typedef struct _snap_shot_data{

        } snap_shot_data;
        
        //needs to be thread safe
        snap_shot_data _snap_data;

        std::function<void()> snap_shot()
        {
            //purpose of this function to update the _snap_data data structure in a threadsafe way
            return [](){};
        }

        template<typename NumInputs, typename B, typename I, typename Kernel>
        void execute(auto trans_sub_id, Kernel kernel, NumInputs, std::function<void()> context, 
                     B buffers, std::array<size_t, std::tuple_size_v<B> > sizes, I items )
        {
            //std::cout << "Executing " << kernel_id << "..." << std::endl;
            auto arr         = get_array_from_tuple( items );
            
            auto _rt_vars    = runtime_vars{ kernel.get_method(),
                                             kernel.get_method_ovr(),
                                             kernel.get_kernel_details() };
            auto _te_buffers = erase_tuple( buffers, sizes );
	    auto exec_parms  = std::vector<size_t>( arr.begin(), arr.end() );
	    //set transactio information
            _rt_vars.associate_transactions( trans_sub_id );

            _runtimeImpl->execute( _rt_vars, NumInputs::value, _te_buffers, exec_parms, options{}); 
            //money maker: this function will interface with the runime Object
        }

        //does nothing in runtime obj
        template<typename ... Us>
        void _forward_exec( ulong tid, ulong& sa_id, Us ... items) {
            if constexpr (std::is_same_v<NullType, _Upstream>) return;
            else
            {
              if( _upstream ) _upstream->_forward_exec(tid, sa_id, items...);
              else std::cout << "Could not find upstream" << std::endl;    
            }
            
        }


    
};
//////////////////////////////////////////////////////////////Definitions///////////////////////////////////////////////////////////////////////////////////
template<typename Upstream>
ExecObj<Upstream>::ExecObj( Upstream upstream)
{
  _upstream = upstream;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Upstream, typename NumInputs, typename Kernel, typename ... Ts>
SubmitObj<Upstream, NumInputs, Kernel, Ts...>::SubmitObj(Upstream upst, NumInputs, Kernel dynamic_override, std::function<void()> snapshot, Ts... ts)
{
    _upstream = upst;
    if constexpr( _num_inputs  != 0 )
    std::cout << " SubmitObj ctor " << std::endl;
    _override_kernel = dynamic_override;
}

template<typename Upstream, typename NumInputs, typename Kernel, typename ... Ts>
template<std::unsigned_integral ... Us>
ExecObj< SubmitObj<Upstream, NumInputs, Kernel, Ts...> > 
SubmitObj<Upstream, NumInputs, Kernel, Ts...>::exec(Us... items)
{
  std::cout << "Start exec..." << std::endl;
  ulong subaction_id   = 0;
  //create a unique_id and makes sure thier is no conflicting Id
  ulong transaction_id = flash_rt::get_runtime()->create_transaction();
  //execute kernels from root node, forward
  _forward_exec(transaction_id, subaction_id, items...);

  flash_rt::get_runtime()->process_transaction( transaction_id );

  return ExecObj(*this);
}

template<typename Upstream, typename NumInputs, typename Kernel, typename ... Ts>
template<std::unsigned_integral ... Us>
auto SubmitObj<Upstream, NumInputs, Kernel, Ts...>::defer(Us... items)
{
  std::cout << "Defering ... " << std::endl;
  auto func1 = [&]<std::size_t N=std::tuple_size_v<typename Upstream::Registry_t>, typename Indices = std::make_index_sequence<N>>()
  {
      auto func2 = [&]<std::size_t... I >(std::index_sequence<I...> ) 
      {
        return RuntimeObj(_upstream->get_runtime(), *this, _upstream->template get_kernel_definition<I>() ... );
      };
    
    return func2(Indices{}); 
  };
   
  return func1();
}

template<typename Upstream, typename NumInputs, typename Kernel, typename ... Ts>
template<std::unsigned_integral ... Us>
SubmitObj<Upstream, NumInputs, Kernel, Ts...>& SubmitObj<Upstream, NumInputs, Kernel, Ts...>::sizes(Us... sizes)
{
  int i=0; 
 //default all var lendth to one
  _sizes.fill(1);
  //override sidves
  int index[sizeof...(sizes)] = {(_sizes[i]=sizes, ++i)...};

  return *this;
}

template<typename Upstream, typename NumInputs, typename Kernel, typename ... Ts>
template< typename ... Us>
void SubmitObj<Upstream, NumInputs, Kernel, Ts...>::_forward_exec(ulong trans_id, ulong& subaction_id, Us ... items )
{  
  //will call prior forward excute until it gets to the root runtime object
  _upstream->_forward_exec(trans_id, subaction_id, items...);
  //Executing 
  auto trans_sub_id = std::make_pair(trans_id, subaction_id );
  _upstream->execute(trans_sub_id, _override_kernel, std::integral_constant<uint, NumInputs::value>{},
                     _snapshot, _buffers, _sizes, std::make_tuple(items...) );

  std::cout << "subaction id = " << subaction_id << std::endl;
  subaction_id++;
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

template<KernelDECL KD, kernel_t k_type=kernel_t::INT_SRC, typename ... Ts>
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

    kernel_t_decl<k_type> _input_program;
    std::optional<std::string> _kernel_name;

};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////a
