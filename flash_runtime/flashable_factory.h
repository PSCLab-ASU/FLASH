#include <flash_runtime/flash_interface.h>
#include <iostream>
#include <memory>
#include <map>

#pragma once

class FlashableRuntimeFactory
{
  public : 
   
    using FlashableRuntimeInfo = FlashableRuntimeMeta<IFlashableRuntime>;

    FlashableRuntimeFactory() = delete;

    static bool Register(const std::string, FlashableRuntimeInfo);

    static std::optional<FlashableRuntimeInfo> Create( const std::string &);

  private:

    static std::map<std::string, FlashableRuntimeInfo> runtimes; 

};



