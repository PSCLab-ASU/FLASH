#include <memory>
#include <string>
#include <flash_runtime/flashable_factory.h>

//std::map<std::string, FlashableRuntimeFactory::FlashableRuntimeInfo> FlashableRuntimeFactory::runtimes; 

bool FlashableRuntimeFactory::Register( const std::string name, FlashableRuntimeInfo info)
{
  auto& runtimes = GetRuntimeMap();
  if( auto it =runtimes.find( name ); it == runtimes.end() )
  {
    std::cout << "Start: Registering : " << name  << std::endl;
    runtimes.insert(std::make_pair(name, info) );
    return true;
  }
  return false;

} 

auto FlashableRuntimeFactory::Create( const std::string& name ) -> 
std::optional<FlashableRuntimeInfo>
{
  auto& runtimes = GetRuntimeMap();
  if( auto it = runtimes.find( name ); it != runtimes.end() )
    return it->second;
  return {};
}


