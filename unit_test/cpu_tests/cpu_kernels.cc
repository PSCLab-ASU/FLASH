#include <stdio.h>
#include <iostream>

extern size_t get_indices( int );

void elmatmult_generic( float * a, float * b, float * c)
{
  auto x = get_indices(0);
  c[x] = a[x] * b[x];

  return;
}

void elmatdiv_generic( float * a, float * b, float * c)
{
  auto x = get_indices(0); 
  c[x] = a[x] / b[x];

  return;
}

