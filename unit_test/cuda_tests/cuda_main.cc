#include <vector>
#include <flash.h>
#include <boost/align/aligned_allocator.hpp>


using MATMULT = KernelDefinition<"elmatmult_generic", kernel_t::EXT_BIN, float*, float*>; 
using MATDIV  = KernelDefinition<"elmatdiv_generic",  kernel_t::EXT_BIN, float*, float*>; 

template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;

int main(int argc, const char * argv[])
{
    //Design Patterns
    // Lazy execution
    // Builder
    // Lookup
    // Self-registry factory
    size_t sz = 512;
    auto chunk = aligned_vector<float>(6*sz, 2);
    float * A = chunk.data(), *B = A + sz, *C = B + sz;
    float * E = C + sz, *F = E + sz, *G = F + sz;

    RuntimeObj ocrt(flash_rt::get_runtime("NVIDIA_GPU") , MATMULT{ argv[0] }, 
                    MATDIV{argv[0]} );
    //submit
    ocrt.submit(MATMULT{}, A, B, C).sizes(sz,sz,sz).defer(sz, sz, sz)
        .submit(MATDIV{},  C, F, G).sizes(sz,sz,sz).exec(sz, sz, sz);

    
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

