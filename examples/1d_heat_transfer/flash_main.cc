#include <algorithm>
#include <iostream>
#include <vector>

#include <flash.h>


//Adapted for FLASH from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/StructuredGrids/1d_HeatTransfer/src
                                                                           
using HEAT_TRANSFER_K = KernelDefinition<1, "init", kernel_t::EXT_BIN, SortBy<2>, unsigned long,  float*, float* >; 


int main(int argc, const char * argv[])
{
   
    size_t n_points = 1000, n_stages=2, n_iter=100;
    flash_memory<float> arr(n_points+2);
    flash_memory<float> arr_next(n_points+2);

    RuntimeObj ocrt(flash_rt::get_runtime("ALL_CPU") , HEAT_TRANSFER_K{ argv[0] } );

    //Create pipeline
    //first submit is a default kernel 'init', then the
    //the compute_heat method is called with an implicit barrier at
    //transitions of n_stages. There are two stages 0, 1 therefore
    //the compute_heat converts into two distinct kernel launches
    //with the implicit barrier we can absorb the swapping between 
    //arr and arr_next, then that process repeats n_iter times.
    ocrt.submit(HEAT_TRANSFER_K{}, n_points, arr, arr_next ).defer(n_points + 2)
        .submit(HEAT_TRANSFER_K{"compute_heat"}, n_points, arr, arr_next )
        .exec(n_points + 1, n_stages, n_iter);

    //Read new positionaa
    //data() method returns a std::vector and implicitly and transfer the
    //entre device buffer to host
    auto& points = arr.data();
    std::cout << "Points : " << points.size() << std::endl;
    std::ranges::copy(points, std::ostream_iterator<int>(std::cout, " "));

    return 0;
}

