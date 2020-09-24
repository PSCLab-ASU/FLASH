#include <memory>
#include <string>
#include  <flash_runtime/flashable_factory.h>

std::map<std::string, FlashableRuntimeFactory::FlashableRuntimeInfo> FlashableRuntimeFactory::runtimes; 
bool FlashableRuntimeFactory::Register( const std::string name, FlashableRuntimeInfo info)
{
  if( auto it =runtimes.find( name ); it == runtimes.end() )
  {
    runtimes[name] = info;
    return true;
  }
  return false;

} 

auto FlashableRuntimeFactory::Create( const std::string& name ) -> 
std::shared_ptr<RuntimeInterface>
{
  if( auto it = runtimes.find( name ); it != runtimes.end() )
    return it->second();
  return nullptr;
}


