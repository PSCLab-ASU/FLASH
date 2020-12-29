#include <vector>
#include "utils/common.h"

//each element in the vector represents the min and max of a task list per dimension
using part_in_t   = const std::vector<std::pair<size_t, size_t> > &;
//each element in the vector represents the min/max of each buffere element
using part_out_t  = std::vector< std::pair<size_t, size_t> >;
using part_func_t = std::add_pointer<part_out_t(part_in_t)>::type;

///////simple paritioning schemes/////
struct no_partition {};
struct single_element {};
//////////////////////////////////////

template<part_func_t part = nullptr >
struct part_by : public Attr
{
  constexpr part_by(part_func_t new_part_func) :
  _part_func(new_part_func)
  {}

  part_func_t _part_func= part;
};


