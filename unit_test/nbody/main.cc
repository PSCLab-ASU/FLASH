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

void particle_init( unsigned long, float * mass, float ** positions, float ** velocities, float ** accelerations, size_t ** ttable);
void process_particles( unsigned long num_particles,float * mass, float ** positions, 
                        float ** velocities, float ** accelerations, float * energy, size_t ** ttable);

int main(int argc, char * argv[] )
{
  const size_t n_particles=16000, y_stages=2, time_steps=10;
  size_t n = n_particles;
  float energy =0;
  int procs = omp_get_num_procs();
  terminated = false;
  /////////////////////////////////////////////////////////////////////////////////////// 
  auto p_init = functor<decltype(particle_init)>(particle_init);
  auto p_comp = functor<decltype(process_particles)>(process_particles);
  ////////////////////////////////////////////////////////////////////////////////////////

  std::cout << "Hello World : " << procs <<  std::endl; 
  std::vector<float> masses(n);
  std::vector<float *>  positions(n, 0), velocities(n,0), accelerations(n,0);
  std::vector<size_t *> task_table(n*y_stages*time_steps, 0);

  //allocate buffer
  for(auto i : std::views::iota((size_t)0, n) )
  {
     positions[i]     = (float *) new float[3]; 
     velocities[i]    = (float *) new float[3]; 
     accelerations[i] = (float *) new float[3]; 
  }

  
  for(auto i : std::views::iota((size_t)0, n * y_stages * time_steps ) )
  {
    task_table[i] = (size_t *) new size_t[3];
    task_table[i][0] = i % n;
    task_table[i][1] = (size_t) ( (i / n) % y_stages );
    task_table[i][2] = (size_t) ( (i / (y_stages*n ) % time_steps) );
    /*std::cout << "{" << task_table[i][0] << "," <<
                        task_table[i][1] << "," << 
                        task_table[i][2] << "}" << std::endl;*/
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  // record start time
  auto start = std::chrono::system_clock::now();

  fill_work_q( n_particles, 1UL, 1UL); 

  #pragma omp parallel for num_threads(procs)
  for(size_t i=0; i < n_particles; i++)
    p_init(0, masses.data(), (float **) positions.data(), (float **) velocities.data(), 
          (float **) accelerations.data(), (size_t **) task_table.data() ); 

  ///////////////////////////////////////////////
  fill_work_q( n_particles, y_stages, time_steps); 
  size_t ** tt = task_table.data();

  for(int k =0; k < time_steps; k++)
  {
    std::cout << "Time step : " << k << std::endl;

    #pragma omp parallel num_threads(procs)
    {
      #pragma omp for
      for(size_t i=0; i < n_particles; i++)
      {
        p_comp(n, masses.data(), (float **) positions.data(), (float **) velocities.data(), 
              (float **) accelerations.data(), &energy, (size_t **) tt ); 

      }

    }

    p_comp(n, masses.data(), (float **) positions.data(), (float **) velocities.data(), 
           (float **) accelerations.data(), &energy, (size_t **) tt ); 

    tt += n_particles;

  }
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>( elapsed_seconds );
  std::cout << "\nTotal Time : " << time_ms.count() << "\n\n";

  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  for(auto i : std::views::iota((size_t)0, n) )
  {
     delete positions[i]; 
     delete velocities[i]; 
     delete accelerations[i]; 
  }

  
  for(auto i : std::views::iota((size_t)0, task_table.size() ) )
    delete task_table[i];

  return 0;
}
