#include <concepts>
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
#include <vector>


struct loop_var
{
  loop_var( unsigned int idx ) :
    _idx(idx),
    _var( std::make_shared<size_t>() ) {}
    
  size_t value()
  {  
    return *_var.get();
  }

  size_t* get()
  {
    return _var.get();
  }

  operator bool()
  {
    return *_var != 0;
  }

  void operator++()
  {
    (*_var)++;
  }

  void operator--()
  {
    (*_var)++;
  }
  unsigned int _idx;
  std::shared_ptr<size_t> _var;
};

class custom_loop
{
  using loop_definition = std::array<size_t, 3>;
  public: 

    custom_loop() {}

    template<std::unsigned_integral ... Ts>
    custom_loop( Ts... ts) {}

    void add_loop_definition( unsigned int idx, size_t stride, size_t offset, size_t val)
    {
      //loop bounds definition
      loop_definition ld = { stride, offset, val }; 
      _loop_array.push_back( ld );

      //loop count
      _lvs.emplace_back(idx);
    }

    custom_loop get_base()
    {
      return *this;
    }

    std::vector<loop_var> _lvs;
    std::vector<loop_definition> _loop_array;
};

class cascade_loop : public custom_loop
{
  public:
 
    //default = max stride = all the subactions
    //          offset = 1
    //          stride = max_stride - offset
    template<typename ... Ts>
    cascade_loop(size_t max_stride,  Ts... ts)
    {
      unsigned int i = 0;
      size_t stride = max_stride;    
      unsigned int _t[sizeof...(Ts)] = { (add_loop_definition( i, stride-i, i, ts), i++ )...};
    }

    template<typename ... Ts>
    cascade_loop(size_t max_stride, unsigned int offset,  Ts... ts)
    {
      unsigned int idx = 0;
      size_t stride = max_stride;
      unsigned int _t[sizeof...(Ts)] = { (add_loop_definition( idx, stride-offset, offset*idx, ts), idx++)... };
    }


};

class seq_loop : public custom_loop
{
  public:

    template<typename ... Ts>
    seq_loop( Ts... ts)
    :  custom_loop( std::forward<Ts>(ts)... ){}
};

struct loop_curator
{
  public :

    bool should_continue(unsigned int loop_idx)
    {
      return false;
    }

    void update( unsigned int loop_idx)
    {

    }

    void add_loops( custom_loop clp)
    {
      _loops = clp;
    }

    custom_loop _loops; 
};
