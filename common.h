#include <vector>
#include <tuple>
#include <memory>
#include <array>
#include <functional>


#pragma once 
enum struct kernel_t { INT_SRC, EXT_SRC, INT_BIN, EXT_BIN }; 

template<kernel_t k_type=kernel_t::INT_SRC, typename T=std::string, typename R=void, typename ... Args>
using kernel_t_decl = std::optional< std::tuple_element_t<(uint)k_type, std::tuple<T, T, 
                                                          std::function<R(Args...)>, T> > >;

struct status {};

struct te_variable
{
  void * data;
  uint type_size;
  size_t vec_size; 

};


template<typename ... Ts, size_t N = sizeof...(Ts), typename Indices = std::make_index_sequence<N> >
std::vector<te_variable> erase_tuple( std::tuple<Ts...> tup,  std::array<size_t, N> sizes )
{
  std::vector<te_variable> _te_vars;
  using tuple_type = std::tuple<Ts...>;

  auto fill = [&]<std::size_t... I >(std::index_sequence<I...> )
  {
    bool dummy[N] ={ (_te_vars.push_back({(void *) std::get<I>(tup), 
                      sizeof(std::tuple_element_t<I,tuple_type>),
                      sizes[I]} ), false)...};
                       
  }; 

  fill(Indices{});

  return  _te_vars;
}

