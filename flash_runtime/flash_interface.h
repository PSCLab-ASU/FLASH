#include <memory>
#include <string>
#include <map>
#include <common.h>

#pragma once

class IFlashableRuntime
{
  public:

    virtual status register_kernels(size_t, kernel_t [], std::string[], 
                                    std::optional<std::string> [] ) = 0;

    virtual status execute(std::string kernel_name, uint num_of_inputs,
                           std::vector<te_variable> kernel_args, 
                           std::vector<te_variable> exec_parms) = 0;
};


template<typename T>
struct FlashableRuntimeMeta
{
  using type = T;

  using IGetMethod = std::shared_ptr<T>(*)();

  void set_creation( IGetMethod method ) { m_GetRuntime = method; }
  void set_description( std::string desc ) { m_Description = desc; }
  std::string get_description () { return m_Description; }

  std::shared_ptr<T> operator()(){ return m_GetRuntime(); };

  IGetMethod  m_GetRuntime;
  std::string m_Description;
};
