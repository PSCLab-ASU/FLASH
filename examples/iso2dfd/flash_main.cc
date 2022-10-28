#include <iostream>
#include <vector>
#include <flash.h>

//Adapted for FLASH from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/N-BodyMethods/Nbody
//in   //in    //inout   //inout  //inout

#define DT 0.002f
#define DXY 20.0f
#define HALF_LENGTH 1
#define BLOCK_SIZE 16

using PROCESS_ISO2D = KernelDefinition<3, "iso_2dfd_kernel", 
                      kernel_t::EXT_BIN, float, size_t, size_t,
		      float*, const float*, const float* >; 

int main(int argc, const char * argv[])
{
  RuntimeObj ocrt( PROCESS_ISO2D{argv[0]} );

  size_t nRows=16, nCols=16;
  unsigned int nIterations;

  size_t nsize = nRows * nCols;
  float dtDIVdxy = (DT * DT) / (DXY * DXY);

  auto prev = std::vector<float>(nsize, 0);
  auto next = std::vector<float>(nsize, 0);
  auto vel  = std::vector<float>(nsize, 0);

  unsigned int grid_cols = (nCols + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (nRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int block_size = BLOCK_SIZE;

  for (unsigned int k = 0; k < nIterations; k += 2) {
    //    swaps their content at every iteration.
    ocrt.submit(PROCESS_ISO2D{}, dtDIVdxy, nRows, nCols, next, prev, vel )
	.defer(grid_cols, grid_rows, block_size, block_size )
        .submit(PROCESS_ISO2D{}, dtDIVdxy, nRows, nCols, prev, next, vel )
	.exec(grid_cols, grid_rows, block_size, block_size );
  }




  return 0;

}

