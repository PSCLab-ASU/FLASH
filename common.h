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

#pragma once 

using override_kernel_t = std::pair<
                            std::string,
                            std::optional<
                              std::pair<
                                std::optional<std::string>,
                                std::optional<std::string> > > >;


struct runtime_vars
{
  std::string lookup;
  std::optional<std::string> kernel_name_override;
  std::optional<std::string> kernel_impl_override;
  std::string get_lookup(){ return lookup; } 
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
};

template<typename T>
struct unary_equals{

  bool operator()(T v){ return v == _val; }

  T _val;
};

/*template<typename Input, typename Container, typename Output>
struct complex_stage 
{
  
  auto operator()(Input& input)
  {
    return input | std::views::transform(_expansion ) | std::views::transform(_reduction );
  }


  std::function<Container(Input)> _expansion;
  std::function<Ouput(Container)> _reduction;
};
*/

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

