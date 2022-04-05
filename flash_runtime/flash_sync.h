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

  template<size_t ... DDims>
  constexpr sort_by(std::vector<size_t> dims = {DDims...} )
  : Attr( KATTR_SORTBY_ID )
  {

    set_generic_vals( {Dims...} );

    if ( sizeof...(DDims) != N)
    {
      set_generic_vals( dims );
    }

  }

};

template<size_t... Dims>
using SortBy = sort_by<Dims...>;


template<size_t... Dims>
struct group_by : public Attr
{
  static const size_t N = sizeof...(Dims);
  using static_vals_t  = std::array<size_t, N>;
  using dynamic_vals_t = std::vector<size_t>;

  template<size_t ... DDims>
  constexpr group_by(std::vector<size_t> dims = {DDims...} )
  : Attr( KATTR_GROUPBY_ID )
  {

    set_generic_vals( {Dims...} );

    if ( (sizeof...(DDims) != N) && !dims.empty() )
    {
      set_generic_vals( dims );
    }

  }
 
};

template<size_t... Dims>
using GroupBy = group_by<Dims...>;
