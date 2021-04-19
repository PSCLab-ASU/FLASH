#include <stdio.h>
#include <iostream>
#include <random>
#include <math.h>
#include <cmath>


//Adapted from https://github.com/oneapi-src/oneAPI-samples.git
//DirectProgramming/DPC++/N-BodyMethods/Nbody

extern size_t get_indices( int );

void init( unsigned long, float * arg, float * arg_next, size_t ** ttable)
{
  size_t x = get_indices( 0 );

  arg[x] = arg_next[x] = 0;
}

void compute_heat( unsigned long num_p, float * arg, float * arg_next, size_t ** ttable)
{
  size_t work_item_id = get_workitem_idx();
  size_t x = get_indices( 0 );
  float C  = 0.5;

  size_t adj_stage_idx = ttable[work_item_id][1];

  if( adj_stage_idx == 0 )
  {
    if( x == num_p + 1)
      arr_next[x] = arr[x - 1];
    else
      arr_next[x] = C * (arr[x + 1] - 2 * arr[x] + arr[x - 1]) + arr[x];
  }
  else
  {
    //swap operation
    arg[x] = arg_next[x];
  }

}

