#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <flash.h>

using ATTN1 = KernelDefinition<4, "attention_1", kernel_t::EXT_BIN, 
                               int, int, const float *, const float *>; 
using ATTN2 = KernelDefinition<3, "attention_2", kernel_t::EXT_BIN, 
                               int, const float *, const float *>; 
using ATTN3 = KernelDefinition<4, "attention_3", kernel_t::EXT_BIN, 
                               int, int, const float*, const float*>; 

int main(int argc, const char * argv[])
{
    int n = 32; 
    int d = 32;
    int repeat = 1;
    std::vector<float> key   (n*d, 0); 
    std::vector<float> value (n*d, 0);
    std::vector<float> score (n,   0);
    std::vector<float> query (d,   0);
    std::vector<float> output(d,   0);
    std::vector<float> prod  (n,   0);
    float exp_sum=0;

    ulong n_gridx  = (n+255)/256;
    ulong n_blockx = 256;
    ulong d_gridx  = (d+255)/256;

    srand(2);
    for (int i = 0; i < n * d; i++) {
      key[i] = 0.1;
      value[i] = 0.3;
      if (rand() % 2)
        query[i % d] = value[i] + key[i] ;
      else
        query[i % d] = value[i] - key[i] ;
    }

    RuntimeObj ocrt( ATTN1{ argv[0] }, 
		     ATTN2{ argv[0] }, 
		     ATTN3{ argv[0] } );
  
    for (int k = 0; k < repeat; k++) 
    {
      // ocrt.submit(ATTN2{}, n, &exp_sum, prod, score )
      //    .exec(n_gridx, 1UL, 1UL, n_blockx, 1UL, 1UL );
      //submit
      ocrt.submit(ATTN1{}, n, d, key, query, prod, &exp_sum )
            .defer(n_gridx, 1UL, 1UL, n_blockx, 1UL, 1UL )
          .submit(ATTN2{}, n, &exp_sum, prod, score )
            .defer(n_gridx, 1UL, 1UL, n_blockx, 1UL, 1UL )
          .submit(ATTN3{}, n, d, score, value, output )
            .exec(d_gridx, 1UL, 1UL, n_blockx, 1UL, 1UL);
    }

    return 0;
}

