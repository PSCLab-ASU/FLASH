#include <vector>
#include <array>
#include <optional>
#include <memory>
#include <type_traits>
#include "utils/common.h"

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
template<typename T=NullType, size_t N = sizeof(std::remove_pointer_t<std::decay_t<T>>),
         size_t Prefetch=4096, bool Managed=true>
struct flash_memory : public Attr
{

  static constexpr int N_elements = sizeof(T) / N;
  using base_t          = std::remove_pointer_t<std::decay_t<T> >;
  using static_cache_t  = std::array<base_t, Prefetch>;
  using dynamic_cache_t = std::vector<base_t>;

  template<IsPointer U>
  constexpr flash_memory(size_t size,
                         U buffer,
                         size_t prefetch = Prefetch )
  : Attr(KATTR_FMEM_ID), _size( size ), _prefetch( prefetch ), _buffer(buffer)
  {
    _app_handle = random_number();
  }

  constexpr flash_memory(size_t size,
                          std::shared_ptr<base_t> buffer,
                          size_t prefetch = Prefetch )
  : Attr(KATTR_FMEM_ID)
  {
    _flash_memory( size, buffer, prefetch );
    _owner = false;
  }

  constexpr flash_memory(size_t size,
                         size_t prefetch )
  : Attr(KATTR_FMEM_ID)
  {
    _flash_memory( size, nullptr, prefetch );
  }

  constexpr flash_memory(size_t size =0 )
  : Attr(KATTR_FMEM_ID)
  {
    /* creates complete buffer */
    _flash_memory( size, nullptr, Prefetch );
    
  }

  //size_t get_id() { return _app_handle; }
  std::string get_id() { return std::to_string(_app_handle); }

  size_t get_type_size() { 
    if( std::is_same_v<T, NullType> )
      return 1;
    else
      return sizeof( T ); 
  }

  bool is_temporary() { return _temporary; }
  size_t get_prefetch_size (){ return _prefetch; }

  //get the host buffer
  auto data()
  {
    return _buffer.get();
    //if constexpr( N_elements == 1) return _buffer.get();
    //else return (T *) _buffer.get();
  }

  size_t size(){ return _size; }

  //get prefetch buffer
  base_t* get_prefetch_data() {
    base_t * out;
    std::visit([&](auto cache) {
      out = reinterpret_cast<base_t *>(cache.data());  
    }, _cache);
    return out;
  }

  flash_memory<T, Prefetch>& id( size_t id ) &&
  {
    _app_handle_ovr = id;
    return *this;
  }

  private:

    constexpr void _flash_memory(size_t size,
                                std::shared_ptr<base_t> buffer,
                                size_t prefetch )
    { 
      printf("Ctor'ing flash_memeory...\n");

      _size     = size;
      _prefetch = prefetch;
      _owner    = true;
      _app_handle = random_number();
      printf("--Mark A...\n");

      if( buffer ){
        printf("--Mark B1...\n");
        _buffer   = buffer;
      }
      else{
        printf("--Mark B2...\n");
        _buffer   = std::shared_ptr<base_t>( (base_t *) malloc( size*sizeof(T)) );
      }

      if( prefetch <= Prefetch){
         printf("--Mark C1...\n");
         /* use static buffer */   
        _cache = static_cache_t();  
      }
      else {
        printf("--Mark C2...\n");
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
    std::shared_ptr< base_t > _buffer;

    std::variant< static_cache_t, dynamic_cache_t> _cache;
};

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

/*constexpr flash_memory operator"" _FM ( unsigned long long handle )
{
    return flash_memory{ handle };
}
*/
