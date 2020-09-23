#include <functionalizer.h>
#include <opencl_runtime/oclrt.h>



using MATMULT = KernelDefinition<"MATMULT", kernel_t::INT_SRC, float*, float*>; 
using MATADD  = KernelDefinition<"MATADD",  kernel_t::INT_SRC, float*, float*>; 
using MATSUB  = KernelDefinition<"MATSUB",  kernel_t::INT_SRC, float*, float*>;

int main(int argc, const char * argv[])
{
    //Design Patterns
    // Lazy execution
    // Builder
    float * A, *B, *C;
    RuntimeObj ocrt(oclrt_runtime::get_runtime() , MATMULT{}, MATADD{}, MATSUB{} );
    //submit
    //ocrt.submit(MATMULT{}, A, B, C).sizes(25UL, 25UL, 25UL).exec(25UL, 25UL, 25UL);       //work items
    ocrt.submit(MATMULT{}, A, B, C).sizes(25UL, 25UL, 25UL).defer(25UL, 25UL, 25UL).
         submit(MATADD{}, A, B, C).sizes(25UL, 25UL, 25UL).defer(25UL, 25UL, 25UL).
         submit(MATSUB{}, A, B, C).sizes(25UL, 25UL, 25UL).exec(25UL, 25UL, 25UL);
  
    return 0;
}

