#include <vector>
#include <tuple>
#include <memory>
#include <array>
#include <set>
#include <functional>
#include <string_view>
#include <optional>
#include <ranges>
#include <algorithm>
#include <climits>
#include <random>
#include <iostream>
#include <type_traits>
#include <cmath>

#pragma once 
#define EXPORT __attribute__((visibility("default")))

struct NullType {};
struct FlashIgnore {};

enum {KATTR_SORTBY_ID=0, KATTR_GROUPBY_ID, KATTR_FMEM_ID };

struct Attr {
  
  Attr(int id ) : _attrId(id) {}

  int get_id(){ return _attrId;}

  void set_generic_vals(  std::vector<size_t> new_vals )
  {
    _generic_values = new_vals;
  }

  std::vector<size_t> get_gen_vals() { return _generic_values; }

  int _attrId;

  std::vector<size_t> _generic_values;

};

enum struct DIRECTION { IN, INOUT, OUT };
enum struct MEM_MOVE { TO_DEVICE, TO_HOST };


struct ParmAttr{
  bool is_pointer;
  bool is_container;
  bool is_flash_mem;
  bool is_rvalue_refer;

  DIRECTION dir;
};

template< typename T>
concept IsAttribute = std::is_base_of_v<Attr, T>;

template<typename T, typename U=std::remove_reference_t<T> >
concept IsContainer = requires (U a){
  typename U::value_type;
  { a.data() } -> std::same_as<std::add_pointer_t<std::remove_reference_t<typename U::value_type> > >;
  a.size();
};


template<typename T>
concept IsPointer = std::is_pointer_v<std::decay_t<T> > &&
(std::is_integral_v<std::remove_pointer_t<std::decay_t<T> > > ||
 std::is_floating_point_v<std::remove_pointer_t<std::decay_t<T> > > );



using override_kernel_t = std::pair<
                            std::string,
                            std::optional<
                              std::pair<
                                std::optional<std::string>,
                                std::optional<std::string> > > >;

template<size_t I>
using expand_void = void *;

struct te_attr;

struct runtime_vars
{
  std::string lookup;
  std::optional<std::string> kernel_name_override;
  std::optional<std::string> kernel_impl_override;
  std::vector<te_attr> kAttrs;
  std::pair<ulong, ulong> trans_subaction_id;

  std::string get_lookup(){ return lookup; } 
  std::pair<ulong, ulong> get_ids() { return trans_subaction_id; }

  void associate_transactions( auto trans_sa_id )
  {
    trans_subaction_id = trans_sa_id;
  }

};


enum struct kernel_t { INT_SRC, EXT_SRC, INT_BIN, EXT_BIN }; 

template<kernel_t k_type=kernel_t::INT_SRC, typename T=std::string, typename R=void, typename ... Args>
using kernel_t_decl = std::optional< std::tuple_element_t<(uint)k_type, std::tuple<T, T, 
                                                          T, T> > >;
struct kernel_desc
{
  kernel_t    _kernel_type;
  std::string _kernel_name;
  std::optional<std::string> _kernel_definition;

  std::string get_kernel_name() { return _kernel_name; }
  kernel_t    get_kernel_type() { return _kernel_type; }
  std::optional<std::string> get_kernel_def(){ return _kernel_definition; }

};

template<typename T>
struct unary_equals{

  bool operator()(T v){ return v == _val; }

  T _val;
};

template<typename T>
struct unary_diff{

  bool operator()(T v){ return v != _val; }

  T _val;
};

struct status {
  int err;
  std::optional<ulong> work_id;

  operator bool()
  {
    return err == 0;
  }

};

struct flash_variable
{
  size_t buffer_id;
  //if true owner of the flash, false if the applications owns
  bool temporary;
  std::pair<void *, size_t> prefetch_buffer;

};


struct te_variable
{
  void * data;
  uint type_size;
  size_t vec_size; 
  ParmAttr parm_attr;
  std::optional<flash_variable> flash_buffer_attr;

  void * get_data()  { return data; }
  size_t get_bytes() { return type_size*vec_size; }
};


struct te_attr
{
  int id;
  std::vector<size_t> dims;
  void(*part)( std::vector<size_t> );
  
};


template<typename ... Ts, size_t N = sizeof...(Ts), typename Indices = std::make_index_sequence<N> >
std::vector<te_variable> erase_tuple( std::tuple<Ts...>& tup,  std::array<ParmAttr, N> parm_attr, 
                                      std::array<size_t, N> sizes )
{
  /* important note: tup is being passed by reference becayse the parameters are being converted to pointers
   * and must live after this function retruns */  
  std::vector<te_variable> _te_vars;
  using tuple_type = std::tuple<Ts...>;

  auto fill = [&]<size_t... I >(std::index_sequence<I...> )
  {
    auto te_conv = [&](size_t Idx, auto&& arg  )
    {
      const bool is_flash_mem = IsAttribute<std::remove_pointer_t<decltype(arg)> >;
      te_variable te; 
      if constexpr ( !is_flash_mem )
      { 
        using Arg = std::remove_pointer_t<decltype(arg) >;

        //primitive data types
        if constexpr( IsContainer< Arg > )
        {
          te = te_variable { reinterpret_cast<void *>( arg->data() ),
                             sizeof(Arg::value_type),
                             arg.size(), parm_attr[Idx], {} };
        }
        else
        {
          te = te_variable { reinterpret_cast<void *>( arg ),
                             sizeof(std::remove_pointer_t<decltype(arg)>),
                             sizes[Idx], parm_attr[Idx], {} };
        }
      }
      else
      {
        //its flash memory
        auto is_temp = arg.is_temporary();
        auto fv = flash_variable{ arg.get_id(), is_temp, 
                                  std::make_pair(arg.get_prefetch_data(), arg.get_prefetch_size() )
                                };
     
        te = te_variable { reinterpret_cast<void *>( arg.data() ), arg.get_type_size(),
                           sizes[Idx], parm_attr[Idx], fv };
        
      }
      
      return te;
     
    };
    
    _te_vars = std::vector{ (te_conv(I, std::get<I>(tup) ))... };
                      
  }; 

  fill(Indices{});

  return  _te_vars;
}

template <typename Container>
bool subsearch(const Container& cont, const std::string& s)
{
    return std::search(cont.begin(), cont.end(), s.begin(), s.end()) != cont.end();
}

template <class S>
auto powerset(const S& s)
{
    std::set<S> ret;
    ret.emplace();
    for (auto&& e: s) {
        std::set<S> rs;
        for (auto x: ret) {
            x.insert(e);
            rs.insert(x);
        }

        ret.insert(std::begin(rs), std::end(rs));
    }

    //erase the first element
    ret.erase( ret.begin() );

    return ret;
} 

template<typename T>
constexpr auto get_array_from_tuple(T&& tuple)
{
  constexpr auto get_array = [](auto&& ... x){ return std::array{std::forward<decltype(x)>(x) ... }; };
  return std::apply(get_array, std::forward<T>(tuple));
}

inline ulong random_number()
{
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<ulong> distrib(0, ULONG_MAX);
  ulong num = distrib(gen);
  return num;
}

template<size_t N, typename Indices = std::make_index_sequence<N> >
void execute_ninput_method( auto& func_ptr, auto& inputs )
{

  auto expand_exec = [&]<size_t... I >(std::index_sequence<I...> )
  {
    auto exec = (void(*)(expand_void<I>...)) func_ptr;
    exec( (inputs[I].data)... );
                       
  }; 

  expand_exec( Indices{} );

}

