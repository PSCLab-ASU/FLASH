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

#pragma once 
#define EXPORT __attribute__((visibility("default")))

enum struct MEM_MOVE { TO_DEVICE, TO_HOST };

using override_kernel_t = std::pair<
                            std::string,
                            std::optional<
                              std::pair<
                                std::optional<std::string>,
                                std::optional<std::string> > > >;

template<size_t I>
using expand_void = void *;

struct runtime_vars
{
  std::string lookup;
  std::optional<std::string> kernel_name_override;
  std::optional<std::string> kernel_impl_override;
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
                                                          std::function<R(Args...)>, T> > >;
struct kernel_desc
{
  kernel_t    _kernel_type;
  std::string _kernel_name;
  std::optional<std::string> _kernel_definition;

  std::string get_kernel_name() { return _kernel_name; }
  std::optional<std::string> get_kernel_def(){ return _kernel_definition; }

};

template<typename T>
struct unary_equals{

  bool operator()(T v){ return v == _val; }

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

struct te_variable
{
  void * data;
  uint type_size;
  size_t vec_size; 

  void * get_data()  { return data; }
  size_t get_bytes() { return type_size*vec_size; }
};


template<typename ... Ts, size_t N = sizeof...(Ts), typename Indices = std::make_index_sequence<N> >
std::vector<te_variable> erase_tuple( std::tuple<Ts...> tup,  std::array<size_t, N> sizes )
{
  std::vector<te_variable> _te_vars;
  using tuple_type = std::tuple<Ts...>;

  auto fill = [&]<std::size_t... I >(std::index_sequence<I...> )
  {
    bool dummy[N] ={ (_te_vars.push_back({(void *) std::get<I>(tup), 
                      sizeof(std::remove_pointer_t<std::tuple_element_t<I,tuple_type> >),
                      sizes[I]} ), false)...};
                       
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
