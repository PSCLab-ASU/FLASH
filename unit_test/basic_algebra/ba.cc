#include <vector>
#include <flash.h>
#include <boost/align/aligned_allocator.hpp>

template <typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, 64>>;


void elwise_matmult( float *, float *, float * )
{
  std::cout <<"--------Calling CPU elwise_matmult--------" << std::endl;
}


//Kernel Defintions
using MATMULT = KernelDefinition<3, "elwise_matmult", kernel_t::EXT_BIN, float*, float*>; 


void host_proxy( std::string backend, std::string impl )
{
    size_t sz = 512;
    auto chunk = aligned_vector<float>(6*sz, 2);
    float * A = chunk.data(), *B = A + sz, *C = B + sz;
    float * E = C + sz, *F = E + sz, *G = F + sz;

    RuntimeObj ocrt(flash_rt::get_runtime(backend), MATMULT{ impl } );

    auto sub = ocrt.submit(MATMULT{}, A, B, C).sizes(sz,sz,sz).defer((size_t)1, (size_t)1, (size_t)1);

    //Processing one element at a time
    //automaitc load balancing across accelerators
    for(int i=1; i < sz; i++)
      //Building mini graph pipeline
      sub = sub.submit(MATMULT{}, A+i, B+i, C+i).sizes(sz,sz,sz).defer((size_t)1, (size_t)1, (size_t)1);

    sub = sub.submit(MATMULT{}, A+sz-1, B+sz-1, C+sz-1).sizes(sz,sz,sz).exec((size_t)1, (size_t)1, (size_t)1);
    
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

}



int main(int argc, const char * argv[])
{
  //uses the current exec (could also use .so)
  std::cout << "Running CPU... : " << std::string(argv[0]) << std::endl;	
  host_proxy("ALL_CPU", argv[0] );

  //uses cubin file
  std::cout << "Running CUDA_RT GPU... : " << std::string(argv[1]) << std::endl;	
  host_proxy("NVIDIA_GPU", argv[1] );

  //uses the aocx file
  std::cout << "Running OpenCL FPGA... : " << std::string(argv[2]) <<  std::endl;	
  host_proxy("INTEL_FPGA", argv[2] );

  return 0;
}

