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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////API relies on Runtime implementation thread-safety, and singleton//////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct NullType {};

//RuntimeObj, SubmitObj, ExecObj, SnapShot
template<typename Upstream, typename NumInputs, typename ... Ts>
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

template<typename Upstream, typename NumInputs, typename ... Ts>
struct SubmitObj
{
  using UpstreamUpstream_t = Upstream::Upstream_t;
  using UpstreamRegistry_t = Upstream::Registry_t;

  SubmitObj(Upstream upst, NumInputs, std::string key, std::function<void()> snapshot, Ts... ts);

  template<std::unsigned_integral ... Us>
  SubmitObj<Upstream, NumInputs, Ts...>& sizes(Us... sizes);

  template<std::unsigned_integral ... Us>
  ExecObj< SubmitObj<Upstream, NumInputs, Ts...> > exec(Us... items);

  template<std::unsigned_integral ... Us >
  auto defer(Us... items);

  //template< ... Us>
  //SubmitObj<Upstream, NumInputs, Ts...>& copy(Us... sizes);
  

  private : 

    template<typename RuntimeImpl, typename _Upstream, typename ... Us>
    friend class RuntimeObj;

    //start forward execution path
    template<typename ... Us>
    void _forward_exec(Us ... );

    template<typename T> friend class ExecObj;

    std::string _kernel_key;
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
    using Upstream_t = _Upstream;
    using Registry_t = std::tuple<Ts...>;

    //constructor without upstream
    explicit  RuntimeObj(RuntimeImpl impl, Ts&& ... ts)
    {
      std::cout << "Calling RuntimeObj without Upstream..." << std::endl;
      auto var    = std::make_tuple(ts...);
      auto entry  = std::get<2>(var);
      auto method = entry.get_method();
      //std::cout << "method = " << method << std::endl;
      _runtimeImpl = impl;
    }

    template< typename InpTL, typename ... Inputs, typename... Outputs, 
          std::size_t N=std::tuple_size_v<typename InpTL::input_ts>, typename Indices = std::make_index_sequence<N>>
    auto submit(InpTL lookup, Inputs... ins, Outputs... outs)
    {
        auto func2 = [&]<std::size_t... I >(std::index_sequence<I...> )
        {
            auto func3 = [&](auto... ins, auto... outs ){


            };
            func3.template operator()<std::tuple_element_t<I, typename InpTL::input_ts>...>(std::forward<Inputs>(ins)..., std::forward<Outputs>(outs)...);
        };
        func2(Indices{}); 

        return SubmitObj(*this, std::integral_constant<size_t, N>{},  lookup.get_method(), snap_shot() );
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

        template<typename Upstream, typename NumInputs, typename ... Us>
        friend struct SubmitObj;

        //constructor with upstream
        RuntimeObj(RuntimeImpl impl, _Upstream& upst, Ts&& ... ts)
        {
          std::cout << "Calling RuntimeObj with Upstream..." << std::endl;
          _upstream = upst;
          _runtimeImpl = impl;

          
        }

        RuntimeImpl get_runtime(){ return _runtimeImpl; }

        //This is incase a deferment occurs
        //std::optional<Upstream> _upstream;
        std::optional<_Upstream> _upstream;
        RuntimeImpl _runtimeImpl;

        typedef struct _snap_shot_data{

        } snap_shot_data;
        
        //needs to be thread safe
        snap_shot_data _snap_data;

        std::function<void()> snap_shot()
        {
            //purpose of this function to update the _snap_data data structure in a threadsafe way
            return [](){};
        }

        template<typename NumInputs, typename B, typename I>
        void execute(std::string kernel_id, NumInputs, std::function<void()> context, 
                     B buffers, std::array<size_t, std::tuple_size_v<B> > sizes, I items )
        {
            std::cout << "Executing " << kernel_id << "..." << std::endl;
            std::array<size_t, std::tuple_size_v<I> > item_sizes;
            item_sizes.fill(1);

            auto _te_buffers = erase_tuple( buffers, sizes );
            auto _te_items   = erase_tuple( items, item_sizes ); 
            _runtimeImpl->execute( kernel_id, NumInputs::value, _te_buffers, _te_items); 
            //money maker: this function will interface with the runime Object
        }

        //does nothing in runtime obj
        template<typename ... Us>
        void _forward_exec(Us ... items) {
            if constexpr (std::is_same_v<NullType, _Upstream>) return;
            else
            {
              if( _upstream ) _upstream->_forward_exec();
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

template<typename Upstream, typename NumInputs, typename ... Ts>
SubmitObj<Upstream, NumInputs, Ts...>::SubmitObj(Upstream upst, NumInputs, std::string key, std::function<void()> snapshot, Ts... ts)
{
    _upstream = upst;
    if constexpr( _num_inputs  != 0 )
    std::cout << " SubmitObj ctor " << key << std::endl;
    _kernel_key = key;
}

template<typename Upstream, typename NumInputs, typename ... Ts>
template<std::unsigned_integral ... Us>
ExecObj< SubmitObj<Upstream, NumInputs, Ts...> > 
SubmitObj<Upstream, NumInputs, Ts...>::exec(Us... items)
{
    std::cout << "Start exec..." << std::endl; 
  //execute kernels from root node, forward
  _forward_exec(items...);

  return ExecObj(*this);
}

template<typename Upstream, typename NumInputs, typename ... Ts>
template<std::unsigned_integral ... Us>
auto SubmitObj<Upstream, NumInputs, Ts...>::defer(Us... items)
{
  std::cout << "Defering ... " << std::endl;
  auto func1 = [&]<std::size_t N=std::tuple_size_v<typename Upstream::Registry_t>, typename Indices = std::make_index_sequence<N>>()
  {
      auto func2 = [&]<std::size_t... I >(std::index_sequence<I...> ) 
      {
        return RuntimeObj(_upstream->get_runtime(), *this, std::tuple_element_t<I, typename Upstream::Registry_t>{}...); 
      };
    
    return func2(Indices{}); 
  };
   
  return func1();
}

template<typename Upstream, typename NumInputs, typename ... Ts>
template<std::unsigned_integral ... Us>
SubmitObj<Upstream, NumInputs, Ts...>& SubmitObj<Upstream, NumInputs, Ts...>::sizes(Us... sizes)
{
  
  return *this;
}

template<typename Upstream, typename NumInputs, typename ... Ts>
template< typename ... Us>
void SubmitObj<Upstream, NumInputs, Ts...>::_forward_exec(Us ... items )
{  
  //will call prior forward excute until it gets to the root runtime object
  _upstream->_forward_exec(items...);
  //Executing 
  _upstream->execute(_kernel_key, std::integral_constant<uint, NumInputs::value>{},
                     _snapshot, _buffers, _sizes, std::make_tuple(items...) );


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
    using program_t = kernel_t_decl<k_type>;
    
    constexpr std::string get_method(){ return std::string(KD.p); }

    KernelDefinition ( kernel_t_decl<k_type> inp={} )
    : _input_program(inp){}

    kernel_t_decl<k_type> _input_program;

};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////a
