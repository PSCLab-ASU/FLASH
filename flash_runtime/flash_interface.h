#include <memory>
#include <string>
#include <map>


#pragma once

class IFlashableRuntime
{
  
  int i=0;
};


template<typename T>
struct FlashableRuntimeMeta
{
  using type = T;

  using IGetMethod = std::shared_ptr<T>(*)();

  void set_creation( IGetMethod method ) { m_GetRuntime = method; }
  void set_description( std::string desc ) { m_Description = desc; }
  std::shared_ptr<T> operator()(){ return m_GetRuntime(); };

  IGetMethod  m_GetRuntime;
  std::string m_Description;
};
