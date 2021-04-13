#include flash_api
#include accelerator_1_api


using Dims    = std::initializer_list<int>;
using strings = std::vector<std::string>;
using lambdas = std::vector<std::function<void(Dims)>>;

void init( std::string backend)
{
  inititalize backend...
}

void init_kernels(strings kernels)
{
  initialize kernels
}


template<typename ... Ts, typename ...Us, typename Ins, typename Outs>
auto prepare_args( Ins ins, Outs outs)
{
  //Heap allocated parameters 

}

template<typename ... Ts, typename ...Us, typename Ins, typename Outs>
void launch_kernel(std::string kernel_name, Ins<Ts...> ins, Outs<Us...> outs)
{
  //launch kernel....
}
 
lambdas construct_pipeline(strings kernel_names)
{
  
  for each kernel_name in kernel_names
  {
    switch case kernel:
      input_tuple = make_tuple(Arg1, Arg2, ArgN...);
      output_tuple = make_tuple(ArgN, ArgN+1, ArgN+M...);
      prepare_args(input_tuple, output_tuple);

      auto exec = [&]( Dims dims ) 
      {
        launch_kernel( input_tuple, output_tuple, dims);

        process_output(kernel_name, out);

        dealloc( kernel, ins, out );
      }; 

     lambdas.push_back( exec );         
  }
}

template<typename Outs>
void process_output( string kernel_name, Outs outs)
{
  switch on kernel_name
    process output
}


int classic_main(int argc, char * argv[] )
{
  ///////////////////////////////////////////////////
  //classsic approach: 
  ////////////////////////////////////////////////////
  init();
  init_kernels();

  strings kernels = {kernel1, kernel2...}; /* define kernel pipeline by name */ 
  auto stages = construct_pipeline(kernels);

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
int flash_main( int argc, char * argv[])
{
  RuntimeObj ocrt(flash_rt::get_runtime("ALL_CPU") , 2PARM_DECL{ argv[0] } );
  ocrt.submit(2PARM_DECL{"kernel1"}, Arg1, Arg2, Arg3).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t)1)
      .submit(2PARM_DECL{"kernel2"}, Arg4, Arg5, Arg6, Arg7).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t) 1)
      .submit(2PARM_DECL{"kernel3"}, Arg8, Arg9, Arg10, Arg11, Arg12).sizes(sz,sz,sz).defer((size_t)32, (size_t)1, (size_t) 1);  
      .submit(2PARM_DECL{"kernel2"}, Arg13Container, Arg14ContOffset13, Arg15COntOffset13, Arg16ContOffset13).exec((size_t)32, (size_t)1, (size_t) 1);

      //notice that each kernels shares the sames number of inputs in this example allowing for the use of a single
      //kernel defintion, additionally, notice that if a parameter is a container and each arg is an offset from that base
      //address will try to infer the sizes of each buffer. Furthermore, these Args can be type flash_memory which serves
      //as an opaque handlde to device memory (see paper for details)

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char * argv[] )
{
  classic_main( argc, argv);

  flash_main( argc, argv);

  return 0;
}


