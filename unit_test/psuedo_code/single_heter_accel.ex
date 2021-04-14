#include flash_api
#include accelerator_1_api
#include accelerator_2_api
#include accelerator_3_api
#.
#.
#.
#include accelerator_N_api


using Dims    = std::initializer_list<int>;
using strings = std::vector<std::string>;
using lambdas = std::vector<std::function<void(Dims)>>;

void init( std::string backend)
{
  switch backend
  {
    inititalize backend...
    ...
  }
}

template<typename kernel_handles>
kernel_handles init_kernels(backend, strings kernels)
{
  initialize kernels
  switch pair( backend, kernels)
    for each kernel in kernels
      init kernel
}


template<typename ... Ts, typename ...Us, typename Ins, typename Outs>
auto prepare_args( backend, Ins ins, Outs outs)
{
  switch backend
  //allocat host and device  parameters 
  
}

template<typename ... Ts, typename ...Us, typename Ins, typename Outs>
void launch_kernel(backend, kintfs, std::string kernel_name, Ins<Ts...> ins, Outs<Us...> outs)
{
  switch backend
    //launch kernel with coressponding kernel intf interfaces
    
}
 
lambdas construct_pipeline(backends, kintfs, strings kernel_names)
{
  
  for each backend, kernel_name in pair(backends, kernel_names)
  {
    switch case kernel:
      input_tuple = make_tuple(Arg1, Arg2, ArgN...);
      output_tuple = make_tuple(ArgN, ArgN+1, ArgN+M...);
      prepare_args(backend, input_tuple, output_tuple);

      auto exec = [&]( Dims dims ) 
      {
        launch_kernel( backend, kintf, input_tuple, output_tuple, dims);

        process_output(backend, kintf, kernel_name, kintf, out);

        dealloc( backend, kernel, ins, out );
      }; 

     lambdas.push_back( exec );         
  }
}

template<typename Outs>
void process_output( backend, kintf, string kernel_name, Outs outs)
{
  switch on pair(backend, kernel_name)
    process output using kintf
}

dealloc( backend, kernel, ins, out )
{
  switch on backend
    //deallocate buffer
}

int classic_main(int argc, char * argv[] )
{
  ///////////////////////////////////////////////////
  //classsic approach: 
  ////////////////////////////////////////////////////
  init();

  strings kernels  = {kernel1, kernel2...}; /* define kernel pipeline by name */ 
  strings backends = {cuda, opencl, hip, thrust, mkl, deep_learning}; 

  //This list get worse when fixed function accelerators come into play
  //For performance sake hip may provide better performance than thrust (or vice-verse)
  //and similarly with opencl,
  //the others are device specific highly optimized frameworks, libraires for programmign models.

  cuda_t kernels1   = init_kernels(cuda, kernels);           //programming model
  opencl_t kernels2 = init_kernels(opencl, kernels);         //programming model
  hip_t kernels3    = init_kernels(hip, kernels);            //portability framework
  thrust_t kernels4 = init_kernels(thrust, kernels);         //portability framework 
  mkl_t kernels5    = init_kernels(mkl, kernels);            //CPU accelerator library
  dl_t  kernels6    = init_kernels(deep_learning, kernels);  //deep learning accelerator

  //tuple all kerel interfaces
  auto tup_kintf( kernels1, kernels2, kernels3, kernels4...);   

  auto stages = construct_pipeline(backends, tup_kintf, kernels);

  for i, each stage in enumerate(stages)
  {
    kernel = kernels[i];
    Dims work_items_xyz ={x, y, z };
    stage( work_items_xyz );
    
  }
    

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
using 2PARM_DECL   = KernelDefinition<2, "2parm_input", kernel_t::INT_BIN, float *, float * >;
using MatrixMult   = KernelDefinition<2, "Matrix4x4",   kernel_t::INT_BIN, float *, float * >;

void init_flash_kernels(strings kernels, strings impls )
{
  //this line registers all the kernels and FLASH interepts which kernels are available 
  //on which backend and makes them available
  //you can also pass instead of 'ALL', a backend identifier
  //the IMPORTANT note here is that the registration lives as long as the application is running.
  //Therefore other invocation of the RuntimeObj does not have re-initialize the kernels
  //The MatrixMult is a fix function in which FLASH will query the available backends for a default implementation
  //After the first time in the loop MatrixMult will be discovered, and subsequent loop iteration be ignored.
  
  for each pair(kernels, impls)
    RuntimeObj ocrt(flash_rt::get_runtime("NVIDIA_GPU") , 2PARM_DECL{kernel, impl }, MatrixMult{} ... );

  
}


int flash_main( int argc, char * argv[])
{

  strings kernels = { "kernel1", "kernel2", "kernel3", "kernel4" };
  string  impls   = {argv[0], argv[1], argv[2], argvr[3] };  

  //initializing kernel repo
  init_flash_kernels(kernels, impls);

  RuntimeObj ocrt;
  ocrt.submit(MatrixMult{}, Arg1, Arg2, Arg3).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t)1)
      .submit(2PARM_DECL{"kernel2"}, Arg4, Arg5, Arg6, Arg7).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t) 1)
      .submit(2PARM_DECL{"kernel3"}, Arg8, Arg9, Arg10, Arg11, Arg12).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t) 1);  
      .submit(2PARM_DECL{"kernel4"}, Arg13Container, Arg14ContOffset13, Arg15COntOffset13, Arg16ContOffset13).exec((size_t)32, (size_t)1, (size_t) 1);

  //notice that the RuntimeObj no longer has to register the kernels for a piece of code to use
  //the acceleration.
  //HUGE NOTE: notice that this code is accelerator agnostic! And devoid of any API specific details/constructs
  //Additionally, RuntimeObj's now hold a repository off all launchable kernels in the program for the life of the application
  //the kernels are associated with thier runtimes without application awareness
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char * argv[] )
{
  classic_main( argc, argv);

  flash_main( argc, argv);

  return 0;
}


