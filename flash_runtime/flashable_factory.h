#include <flash_runtime/flash_interface.h>
#include <iostream>
#include <memory>
#include <map>

#pragma once

class FlashableRuntimeFactory
{
  public : 
   
    using FlashableRuntimeInfo = FlashableRuntimeMeta<IFlashableRuntime>;
    using map_type = std::map<std::string, FlashableRuntimeInfo>;

    FlashableRuntimeFactory() = delete;

    static bool Register(const std::string, FlashableRuntimeInfo);

    static std::optional<FlashableRuntimeInfo> Create( const std::string &);

    static std::vector<std::string> List();
  
    static map_type& GetRuntimeMap()
    {
      static map_type runtimes; 
      return runtimes;
    }

  private:

    //static map_type runtimes; 


};
 
