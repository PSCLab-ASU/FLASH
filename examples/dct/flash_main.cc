#include <vector>
#include <flash.h>
#include <dct.h>


using DCT = KernelDefinition<4, "DCT8x8_kernel", kernel_t::EXT_BIN, GroupBy<6>,
                             uint, uint, uint, const  float*, float*>; 

inline unsigned int iDivUp(unsigned int dividend, unsigned int divisor){
    return dividend / divisor + (dividend % divisor != 0);
}

int main(int argc, const char * argv[])
{
    unsigned long imageH=300;
    unsigned long imageW=300;
    unsigned long stride=imageW;
    int dir = DCT_FORWARD;
    unsigned long blockSize[2];
    unsigned long gridSize[2];
    unsigned long nIterations = 16;
    
    std::vector<float> dst(imageH*stride, 0); 
    std::vector<float> src(imageH*stride, 0); 

    blockSize[0] = BLOCK_X;
    blockSize[1] = BLOCK_Y / BLOCK_SIZE;
    gridSize[0]  = iDivUp(imageW, BLOCK_X);
    gridSize[1]  = iDivUp(imageH, BLOCK_Y);

    RuntimeObj ocrt( DCT{ argv[0] } );

    if (dir == DCT_FORWARD)  {
      ocrt.submit(DCT{}, imageH, imageW, stride, src, dst)
	  .exec(gridSize[0], gridSize[1], 1UL, blockSize[0], blockSize[1], nIterations );
    }
    else {
      ocrt.submit(DCT{"IDCT8x8_kernel"}, imageH, imageW, stride, src, dst)
	  .exec(gridSize[0], gridSize[1], 1UL, blockSize[0], blockSize[1], nIterations );
    }

    //submit

    return 0;
}

