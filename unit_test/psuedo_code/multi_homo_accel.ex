#include flash_api
#include accelerator_1_api


using Dims    = std::initializer_list<int>;
using strings = std::vector<std::string>;
using lambdas = std::vector<std::function<void(Dims)>>;

void init( )
{
  inititalize backend...
}

template<typename kernel_handles>
kernel_handles init_kernels(device_qs, strings kernels)
{
  //This function also explodes
  initialize kernels
  for_each valid device_q
    for each kernel in kernels
      init kernel
}


template<typename ... Ts, typename ...Us, typename Ins, typename Outs>
auto prepare_args( kintf, Ins ins, Outs outs)
{
  //this method EXPLODES as well
  switch kintf
  //need device knowledge to allocate buffer on correct device
  //allocat host and device  parameters 
  
}

template<typename ... Ts, typename ...Us, typename Ins, typename Outs>
void launch_kernel(kintfs, std::string kernel_name, Ins<Ts...> ins, Outs<Us...> outs)
{
  //this launch code logic explodes if you had to handle multiple accelerators
  //keeping track of N accelerators and load balancing is a hassle
  decide which kernelintf queue to deploy on
    //launch kernel with coressponding using the corresponding kernel interface( including queue)
    
}
 
lambdas construct_pipeline(kintfs, strings kernel_names)
{
  
  for each kernel_name in kernel_names
  {
    switch case kernel:
      input_tuple = make_tuple(Arg1, Arg2, ArgN...);
      output_tuple = make_tuple(ArgN, ArgN+1, ArgN+M...);
      prepare_args(kintf, input_tuple, output_tuple);

      auto exec = [&]( Dims dims ) 
      {
        launch_kernel(kintf, input_tuple, output_tuple, dims);

        process_output(kintf, kernel_name, kintf, out);

        dealloc( kintf, kernel, ins, out );
      }; 

     lambdas.push_back( exec );         
  }
}

template<typename Outs>
void process_output( kintf, string kernel_name, Outs outs)
{
  //this function does EXPLODE too bad
  switch on kernel_name
    process output using kintf and the device queues

}

dealloc(kintf, kernel, ins, out )
{
  switch on kintf
    //deallocate buffer; includes queues
}

int classic_main(int argc, char * argv[] )
{
  ///////////////////////////////////////////////////
  //classsic approach: 
  ////////////////////////////////////////////////////
  init();

  strings kernels  = {kernel1, kernel2...}; /* define kernel pipeline by name */ 

  //This list get worse when fixed function accelerators come into play
  //For performance sake hip may provide better performance than thrust (or vice-verse)
  //and similarly with opencl,
  //the others are device specific highly optimized frameworks, libraires for programmign models.
  
  devices = get_list_of_accelerators()

  //tuple of vectors of all kernel interfaces
  auto tup_kintf( kernel_t );   

  for each device in devices
    if valid set create queues

  for each valid queue  in queues:
    intitialize kernel interfces...
      kernel_t kernels1   = init_kernels(queue, kernels);

    //push kernel intf to tuple vector
    std::get<kernel_t>(tup_kintf).push_back(kernels1);

  auto stages = construct_pipeline(tup_kintf, kernels);

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
  
  //Notice to target a single type of accelerator only the runtime variable needs to change to
  //target a specific backend

  for each pair(kernels, impls)
    RuntimeObj ocrt(flash_rt::get_runtime("NVIDIA_GPU") , 2PARM_DECL{kernel, impl }, MatrixMult{} ... );

  
}


int flash_main( int argc, char * argv[])
{

  strings kernels = { "kernel1", "kernel2", "kernel3", "kernel4" };
  string  impls   = {argv[0], argv[1], argv[2], argvr[3] };  

  //initializing kernel repo
  //FLASH handles the quantity of any given type of accelerator and LOAD BALANCES accordingly
  init_flash_kernels(kernels, impls);

  RuntimeObj ocrt;
  ocrt.submit(MatrixMult{}, Arg1, Arg2, Arg3).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t)1)
      .submit(2PARM_DECL{"kernel2"}, Arg4, Arg5, Arg6, Arg7).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t) 1)
      .submit(2PARM_DECL{"kernel3"}, Arg8, Arg9, Arg10, Arg11, Arg12).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t) 1);  
      .submit(2PARM_DECL{"kernel4"}, Arg13Container, Arg14ContOffset13, Arg15COntOffset13, Arg16ContOffset13).exec((size_t)32, (size_t)1, (size_t) 1);

  //notice that the RuntimeObj no longer has to register the kernels for a piece of code to use
  //the acceleration.
  //IMPORTANT NOTE: FLASH finds a valid accelator (or sets of accelerators) to run the computation.
  //HUGE NOTE: notice that this code not only maintains accelerator independence, it also brokers device usage transparently.
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


