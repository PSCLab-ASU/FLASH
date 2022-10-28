#include <iostream>
#include <vector>
#include <flash.h>

//in   //in    //inout   //inout  //inout
//__global__ void process_conv(float *A, float *B, float *bnBias, float *bnScale, float *C)

using PROCESS_CONV = KernelDefinition<4, "process_conv", 
                       kernel_t::EXT_BIN, 
		       float*, float*, float*, float*, float* >; 

int main(int argc, const char * argv[])
{
  uint nShm      = (4*512 + 64*128 + 4*128 + 2*128)<<2;
  int  nInput    = (14*14*512)<<3, nOutput = (14*14*128)<<2, nWeights = (128*512)<<2;
  int  nBias     = 128 <<2, nScale = 128 << 2;
  uint nBlockXs  = 128, nBlockYs = 4, nThrdPerBlk = 49; 

  std::vector<float> input (nInput,    0);
  std::vector<float> output(nOutput,   0);
  std::vector<float> weight(nWeights,  0);
  std::vector<float> bias  (nBias,     0);
  std::vector<float> scale (nScale,    0);

  RuntimeObj ocrt( PROCESS_CONV{argv[0]}, PROCESS_CONV{ argv[0] } );

  ocrt.submit(PROCESS_CONV{}, input, weight, bias, scale, output )
      .exec(nBlockXs, nBlockYs, 1U, nThrdPerBlk, 1U, 1U, nShm );

  return 0;

}

