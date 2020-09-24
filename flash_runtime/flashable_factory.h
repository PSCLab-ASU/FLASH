#include <flash_runtime/flash_interface.h>
#include <memory>
#include <map>

#pragma once

class FlashableRuntimeFactory
{
  public : 
   
    using FlashableRuntimeInfo = FlashableRuntimeMeta<IFlashableRuntime>;
    using RuntimeInterface = typename FlashableRuntimeInfo::type;

    FlashableRuntimeFactory() = delete;

    static bool Register(const std::string, FlashableRuntimeInfo);

    static std::shared_ptr<RuntimeInterface> Create( const std::string &);

  private:

    static std::map<std::string, FlashableRuntimeInfo> runtimes; 

};



