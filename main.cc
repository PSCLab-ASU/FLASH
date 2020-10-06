#include <flash.h>



using MATMULT = KernelDefinition<"MATMULT", kernel_t::EXT_BIN, float*, float*>; 
using MATADD  = KernelDefinition<"MATADD",  kernel_t::EXT_BIN, float*, float*>; 
using MATSUB  = KernelDefinition<"MATSUB",  kernel_t::EXT_BIN, float*, float*>;

int main(int argc, const char * argv[])
{
    //Design Patterns
    // Lazy execution
    // Builder
    // Lookup
    // Self-registry factory
    float * A, *B, *C;
    std::string dir = "/archive-t2/Design/fpga_computing/wip/fpga_exmaples/bin/";
    RuntimeObj ocrt(flash_rt::get_runtime("INTEL_FPGA") , 
                    MATMULT{dir + "elwise_matmult_gen.aocx" }, 
                    MATADD {dir + "elwise_matdiv_gen.aocx"  }, 
                    MATSUB {dir + "elwise_matmult_gen.aocx" } );
    //submit
    //ocrt.submit(MATMULT{}, A, B, C).sizes(25UL, 25UL, 25UL).exec(25UL, 25UL, 25UL);       //work items
    ocrt.submit(MATMULT{}, A, B, C).sizes(25UL, 25UL).defer(25UL, 25UL, 25UL).
         submit(MATADD{},  A, B, C).sizes(25UL, 25UL, 25UL).defer(25UL, 25UL, 25UL).
         submit(MATSUB{},  A, B, C).sizes(25UL, 25UL, 25UL).exec(25UL, 25UL, 25UL);
  
    return 0;
}

