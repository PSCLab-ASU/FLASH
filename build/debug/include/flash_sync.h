#include <array>
#include <vector>
#include <variant>
#include <type_traits>
#include "utils/common.h"


template<size_t ... Dims>
struct sort_by : Attr
{
  static const size_t N = sizeof...(Dims);
  using static_dims_t  = std::array<size_t, N>;
  using dynamic_dims_t = std::vector<size_t>;

  template<typename ... Us>
  constexpr sort_by(Us ... dims)
  {
    if constexpr( sizeof...(Us) != N)
    {
      _dims = dynamic_dims_t{dims...};
    }
  }

  //static   
  std::variant<static_dims_t, dynamic_dims_t> _dims 
       = static_dims_t{Dims...};

};



template<uint Dim=0, size_t... Values>
struct group_by : public Attr
{
  static const size_t N = sizeof...(Values);
  using static_vals_t  = std::array<size_t, N>;
  using dynamic_vals_t = std::vector<size_t>;

  template<size_t ... sizes>
  constexpr group_by(uint dim, std::vector<size_t> values = {sizes...} )
  {
    if ( sizeof...(sizes) != N)
    {
      _values = values;
    }

    _dim = dim;
  }
 
  uint _dim = Dim;  

  /* values */
  std::variant<static_vals_t, dynamic_vals_t> _values 
         = static_vals_t{Values...};
  
};

