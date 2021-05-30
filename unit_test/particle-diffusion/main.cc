#include <iostream>
#include <cuda.h>
#include <vector>
#include <queue>
#include <thread>
#include <map>
#include <mutex>
#include <array>
#include <ranges>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <math.h>


struct work_item
{
  size_t wi_id;
  size_t index[3];
};

static bool terminated = false;
static std::mutex g_mutex;
static std::queue<work_item> g_task_q;
static std::map<int, work_item> g_task_map;

size_t get_indices( int idx )
{
  auto id = omp_get_thread_num();
  return g_task_map.at(id).index[idx];
}

size_t get_workitem_idx()
{
  auto id = omp_get_thread_num();
  return g_task_map.at(id).wi_id;
}

void early_terminate()
{
  std::lock_guard<std::mutex> lk(g_mutex);
  terminated = true;
}

template<typename ...Ts>
void fill_work_q(Ts ... szs )
{
  size_t sizes[sizeof...(szs)] = { szs... };
  size_t total_wi =1;
  size_t x, y, z;

  for(int i =0; i < sizeof...(szs); i++)
    total_wi *= sizes[i];

  g_task_q = std::queue<work_item>();

  for(size_t i=0; i < total_wi; i++)
  {
    x = i % sizes[0];
    y = (size_t) ( (i / sizes[0]) % sizes[1] );
    z = (size_t) ( (i / (sizes[1]*sizes[0] ) % sizes[2]) );
    g_task_q.push({i, {x, y, z } } ); 
  }
}

void fill_current_work_idx(int id)
{ 
  std::lock_guard<std::mutex> lk(g_mutex);

  work_item next_wi = g_task_q.front();
  g_task_q.pop();
  g_task_map[id] = std::move(next_wi); 
}

template<typename T>
struct functor
{
  functor( T func ) : _func(func) {};

  template<typename ... Ts>
  void operator()(Ts... ts)
  {
    auto id = omp_get_thread_num();
    fill_current_work_idx( id );
    _func(ts...);
  }

  T* _func;  
};


//kernels
void particle_init(size_t *, float * posX, float * posY, float *, float *, size_t **);
void grid_init( size_t * grid, float *, float *, float *, float *, size_t **);
void random_init( size_t *, float *, float *, float * randX, float * randY, size_t **);
void process_particles( unsigned long grid_size, size_t n_particles, float radius,
                        unsigned int * prev_known_cell_coordinate_XY,
                        size_t * grid_a, float * particle_X_a, float * particle_Y_a,
                        float * random_X_a, float * random_Y_a, size_t ** ttable);



int main(int argc, char * argv[] )
{
  const size_t grid_size=22, planes=3, n_particles=256, n_iter=10000;
  const size_t grid_sz = std::pow(grid_size,2) * planes;
  const size_t nmove = n_particles * n_iter;
  size_t n = n_particles;
  float radius = 0.5f;

  int procs = omp_get_num_procs();
  terminated = false;
  /////////////////////////////////////////////////////////////////////////////////////// 
  auto p_init = functor<decltype(particle_init)>(particle_init);
  auto g_init = functor<decltype(grid_init)>(grid_init);
  auto r_init = functor<decltype(random_init)>(random_init);
  auto p_comp = functor<decltype(process_particles)>(process_particles);
  ////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "Hello World : " << procs <<  std::endl; 
  std::vector<size_t> grid(grid_sz, 0);
  std::vector<size_t *> task_table(nmove, 0);
  std::vector<unsigned int> prevXY( 3*n, 0);
  std::vector<float> posX(n, 0), posY(n, 0), 
                     randX(2*nmove, 0), randY(2*nmove, 0);

  for(auto i : std::views::iota((size_t)0, nmove ) )
  {
    task_table[i] = (size_t *) new size_t[2];
    task_table[i][0] = i % n;
    task_table[i][1] = (size_t) ( (i / n) % n_iter );
    //std::cout << "{" << task_table[i][0] << "," <<
    //                    task_table[i][1] <<  "}" << std::endl;
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  // record start time
  auto start = std::chrono::system_clock::now();

  fill_work_q( n_particles, 1UL); 
  #pragma omp parallel for num_threads(procs)
  for(size_t i=0; i < n_particles; i++)
    p_init( nullptr, posX.data(), posY.data(), nullptr, nullptr, nullptr ); 
 
  fill_work_q( grid_sz, 1UL); 
  #pragma omp parallel for num_threads(procs)
  for(size_t i=0; i < grid_sz; i++)
    g_init( grid.data(), nullptr, nullptr, nullptr, nullptr, nullptr ); 
 
  fill_work_q( nmove, 1UL); 
  #pragma omp parallel for num_threads(procs)
  for(size_t i=0; i < nmove; i++)
    r_init( nullptr, nullptr, nullptr, randX.data(), randY.data(), nullptr ); 

  ///////////////////////////////////////////////
  fill_work_q( n_particles, n_iter); 
  size_t ** tt = task_table.data();

  for(int k =0; k < n_iter; k++)
  {
    #pragma omp parallel num_threads(procs)
    {
      #pragma omp for
      for(size_t i=0; i < n_particles; i++)
      {
        p_comp(grid_size, n_particles, radius, prevXY.data(), 
               grid.data(), posX.data(), posY.data(), 
               randX.data(), randY.data(), (size_t **) tt ); 

      }

    }

    tt += n_particles;

  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>( elapsed_seconds );
  std::cout << "\nTotal Time : " << time_ms.count() << "\n\n";

  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  for(auto i : std::views::iota((size_t)0, task_table.size() ) )
    delete task_table[i];

  return 0;
}
