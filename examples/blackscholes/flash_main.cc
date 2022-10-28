#include <iostream>
#include <vector>
#include <flash.h>
#include <blackScholesAnalyticEngineStructs.h>

//Adapted for FLASH from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/N-BodyMethods/Nbody
//in   //in    //inout   //inout  //inout
//getOutValOption(optionInputStruct* options, float* outputVals, int numVals)

using PROCESS_SCHOL = KernelDefinition<2, "getOutValOption", 
                       kernel_t::EXT_BIN, int, optionInputStruct *, float * >; 

int main(int argc, const char * argv[])
{
  int numVals = 50e6;  
  float * output;
  unsigned long gx   = (numVals + THREAD_BLOCK_SIZE - 1)/THREAD_BLOCK_SIZE;
  unsigned long thdx = THREAD_BLOCK_SIZE;

  std::vector<optionInputStruct> input (numVals);

  prepare_input( input, numVals );
  
  RuntimeObj ocrt( PROCESS_SCHOL{argv[0]} );

  ocrt.submit(PROCESS_SCHOL{}, numVals, input, output )
      .exec(gx, 1UL, 1UL, thdx, 1UL, 1UL );

  return 0;

}

