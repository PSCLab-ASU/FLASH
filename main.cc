#include <flash.h>



using MATMULT = KernelDefinition<"MATMULT", kernel_t::INT_SRC, float*, float*>; 
using MATADD  = KernelDefinition<"MATADD",  kernel_t::INT_SRC, float*, float*>; 
using MATSUB  = KernelDefinition<"MATSUB",  kernel_t::INT_SRC, float*, float*>;

int main(int argc, const char * argv[])
{
    //Design Patterns
    // Lazy execution
    // Builder
    // Lookup
    // Self-registry factory
    float * A, *B, *C;
    RuntimeObj ocrt(flash_rt::get_runtime("INTEL_FPGA") , MATMULT{"matmult.aocx"}, MATADD{"matadd.aocx"}, MATSUB{"matsub.aocx"} );
    //submit
    //ocrt.submit(MATMULT{}, A, B, C).sizes(25UL, 25UL, 25UL).exec(25UL, 25UL, 25UL);       //work items
    ocrt.submit(MATMULT{"matmult_v1"}, A, B, C).sizes(25UL, 25UL, 25UL).defer(25UL, 25UL, 25UL).
         submit(MATADD{"matadd_v1"},   A, B, C).sizes(25UL, 25UL, 25UL).defer(25UL, 25UL, 25UL).
         submit(MATSUB{"matsub_v1"},   A, B, C).sizes(25UL, 25UL, 25UL).exec(25UL, 25UL, 25UL);
  
    return 0;
}

