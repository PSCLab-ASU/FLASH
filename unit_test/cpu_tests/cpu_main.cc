#include <vector>
#include <flash.h>
#include <boost/align/aligned_allocator.hpp>

//extern size_t get_indices( int );

using MATMULT = KernelDefinition<"elmatmult_generic", kernel_t::INT_BIN, float*, float*>; 
using MATDIV  = KernelDefinition<"elmatdiv_generic",  kernel_t::INT_BIN, float*, float*>; 

template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

struct TEST
{
  TEST(int i) : _i(i) {}

  [[gnu::used]]
  void elmatdiv_generic( float *a, float *b, float *c)
  {
    auto x = get_indices(0);
    c[x] = a[x] / b[x] + _i;
    return;
  }

  int _i=0;

};

using MATDIV_T  = KernelDefinition<"TEST::elmatdiv_generic", kernel_t::INT_BIN, TEST*, float*, float*>; 

int main(int argc, const char * argv[])
{
    //Design Patterns
    // Lazy execution
    // Builder
    // Lookup
    // Reflectiona
    // Dynamic dispatching
    // Self-registry factory
    size_t sz = 512;
    TEST t1(33);

    auto chunk = aligned_vector<float>(6*sz, 2);
    float * A = chunk.data(), *B = A + sz, *C = B + sz;
    float * E = C + sz, *F = E + sz, *G = F + sz;

    RuntimeObj ocrt(flash_rt::get_runtime("ALL_CPU") , MATMULT{ argv[0] }, 
                    MATDIV_T{argv[0]} );
    //submit
    ocrt.submit(MATMULT{}, A, B, C).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t)1)
        .submit(MATDIV_T{}, &t1, C, F, G).sizes((size_t) 1, sz,sz,sz).exec((size_t)32, (size_t)1, (size_t) 1);

    
    std::cout << "C = ";
    for(auto i : std::views::iota(0,9) )
    {
      std::cout << C[i] << ",";
    }
    std::cout << C[10] << std::endl;

    std::cout << "G = ";
    for(auto i : std::views::iota(0, 9) )
    {
      std::cout << G[i] << ",";
    }
    std::cout << G[10] << std::endl;
    

    return 0;
}

