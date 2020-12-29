#include <vector>
#include <array>
#include <optional>
#include <memory>
#include <type_traits>
#include "utils/common.h"

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
template<typename T=NullType, size_t Prefetch=4096, bool Managed=true>
struct flash_buffer : public Attr
{

  using static_cache_t  = std::array<T, Prefetch>;
  using dynamic_cache_t = std::vector<T>;
 
 
  template<IsPointer U>
  constexpr flash_buffer(size_t size,
                         U buffer,
                         size_t prefetch = Prefetch )
  : _size( size ), _prefetch( prefetch ), _buffer(buffer)
  {
    _app_handle = random_number();
  }

  constexpr flash_buffer(size_t size,
                          std::shared_ptr<T> buffer,
                          size_t prefetch = Prefetch )
  {
    _flash_buffer( size, buffer, prefetch );
    _owner = false;
  }

  constexpr flash_buffer(size_t size,
                         size_t prefetch )
  {
    _flash_buffer( size, nullptr, prefetch );
  }

  constexpr flash_buffer(size_t size =0 )
  {
    /* creates complete buffer */
    _flash_buffer( size, nullptr, Prefetch );
    
  }

  size_t get_id() { return _app_handle; }

  size_t get_type_size() { 
    if( std::is_same_v<T, NullType> )
      return 1;
    else
      return sizeof( T ); 
  }

  bool is_temporary() { return _temporary; }
  size_t get_prefetch_size (){ return _prefetch; }

  //get the host buffer
  T * data(){ return _buffer.get(); }
  size_t size(){ return _size; }

  //get prefetch buffer
  T * get_prefetch_data() {
    T * out;
    std::visit([&](auto cache) {
      out = cache.data();  
    }, _cache);
    return out;
  }

  flash_buffer<T, Prefetch>& id( size_t id ) &&
  {
    _app_handle_ovr = id;
    return *this;
  }

  private:

    constexpr void _flash_buffer(size_t size,
                                std::shared_ptr<T> buffer,
                                size_t prefetch )
    {
      _size     = size;
      _prefetch = prefetch;
      _buffer   = buffer;
      _owner    = true;
      _app_handle = random_number();

      if( prefetch <= Prefetch){
         /* use static buffer */   
       _cache = static_cache_t();  
      }
      else {
         /* use dynamic buffer */
       _cache = dynamic_cache_t( prefetch );
      }

    }

    bool   _owner = true;
    bool   _temporary = true;
    size_t _app_handle;
    size_t _size;
    size_t _prefetch;
    std::optional<size_t> _app_handle_ovr;
    std::optional<size_t> _flash_handle;
    std::shared_ptr< T > _buffer;
 
    std::variant< static_cache_t, dynamic_cache_t> _cache;
};

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

/*constexpr flash_buffer operator"" _FM ( unsigned long long handle )
{
    return flash_buffer{ handle };
}
*/
