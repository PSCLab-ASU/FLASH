#include <stdio.h>

#define DT 0.002f
#define DXY 20.0f
#define HALF_LENGTH 1
#define BLOCK_SIZE 16

__global__ void iso_2dfd_kernel(const float dtDIVdxy, const int nRows, const int nCols, 
		                float* next, const float* prev, const float* vel )
{
  // Compute global id
  // We can use the get.global.id() function of the item variable
  //   to compute global id. The 2D array is laid out in memory in row major
  //   order.
  int gidCol = blockDim.x * blockIdx.x + threadIdx.x;
  int gidRow = blockDim.y * blockIdx.y + threadIdx.y;
  float value = 0.f;

  if (gidRow < nRows && gidCol < nCols) {

    size_t gid = (gidRow)*nCols + gidCol;

    // Computation to solve wave equation in 2D
    // First check if gid is inside the effective grid (not in halo)
    if ((gidCol >= HALF_LENGTH && gidCol < nCols - HALF_LENGTH) &&
        (gidRow >= HALF_LENGTH && gidRow < nRows - HALF_LENGTH)) {
      // Stencil code to update grid point at position given by global id (gid)
      // New time step for grid point is computed based on the values of the
      //    the immediate neighbors in both the horizontal and vertical
      //    directions, as well as the value of grid point at a previous time step
      value = 0.f;
      value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
      value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
      value *= dtDIVdxy * vel[gid];
      next[gid] = 2.f * prev[gid] - next[gid] + value;
    }
  }
}
