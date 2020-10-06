#include "opencl_runtime/oclrt.h"
#include <iostream>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <tuple>
//#include <experimental/ranges/algorithm>

/* Registers the factory with flash factory*/
bool ocl_runtime::_registered = FlashableRuntimeFactory::Register(
                                ocl_runtime::get_factory_name(),
                                ocl_runtime::get_runtime() );

std::shared_ptr<ocl_runtime> ocl_runtime::_global_ptr;


FlashableRuntimeMeta<IFlashableRuntime> ocl_runtime::get_runtime()
{
  //automatic polymorphism to base classa
  FlashableRuntimeMeta<IFlashableRuntime> out{ (std::shared_ptr<IFlashableRuntime> (*)()) get_singleton, 
                                                get_factory_desc() };

  return out;
}


std::shared_ptr<ocl_runtime> ocl_runtime::get_singleton()
{

  if( _global_ptr ) return _global_ptr;
  else return _global_ptr = std::shared_ptr<ocl_runtime>( new ocl_runtime() );

}

ocl_runtime::ocl_runtime()
{
  std::cout << "Ctor'ing oclrt_runtime...." << std::endl;
  cl_int err;
  //get all the device ids
  auto device_ids = _get_devices();
  auto device_set = std::set(std::begin(*device_ids), std::end(*device_ids) );
  //create a powerset
  auto dev_comb   = powerset(device_set);

  //create all context ccombinationsa 
  std::ranges::for_each(dev_comb, [&](auto dev_set)
  {
    std::cout << "Creating context " << dev_set.size() << std::endl;
    auto device_ss = std::vector(std::begin(dev_set), std::end(dev_set));

    //create contexts 
    auto ctx = clCreateContext( NULL, device_ss.size(), device_ss.data(), NULL, NULL, &err);
    this->_contexts.emplace_back( ctx, device_ss );

  } );

}

status ocl_runtime::execute(runtime_vars rt_vars, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "Executing from opencl_runtime..." << rt_vars.get_lookup() << std::endl;
  std::ranges::for_each(kernel_args, [](auto te_var)
  {
    std::cout << "sizes are : " << te_var.vec_size << std::endl;
  });

  return {};
}

status ocl_runtime::register_kernels( const std::vector<kernel_desc>& kds ) 
{
  
  std::cout << "calling ocl_runtime::" << __func__<<  std::endl;
  auto kern = kernel_t::EXT_BIN;
  
  if( std::ranges::all_of(kds, unary_equals{kern}, &kernel_desc::_kernel_type) )
  {

    auto binaries = _read_kernel_files(kds);

    _append_programs_kernels( binaries );   

  }
  else
  {
    std::cout << "Invalid input args..." << std::endl;
  }

  std::cout << "completed registering kernels" << std::endl;
  return {}; 
}

std::vector<std::tuple<std::string, std::optional<std::string>, std::optional<std::string> > >
ocl_runtime::_read_kernel_files( const std::vector<kernel_desc>& kds)
{
  std::vector<std::tuple<std::string, std::optional<std::string>, 
                         std::optional<std::string> > > binaries;
  //read alll file locations
  std::ranges::transform( kds, std::back_inserter(binaries), 
  [](const kernel_desc& kd)
  {
     auto kernel_name   = kd._kernel_name; 
     auto impl_location = kd._kernel_definition;
     
     if( impl_location )
     {
       //read contents of file
       std::ifstream bin(impl_location.value(), std::ios::binary);

       if( !bin.good() ) 
       {
         std::cout << "Could not locate " << impl_location.value() << std::endl;
         return std::make_tuple(kernel_name, impl_location, std::optional<std::string>{} );
       }
       //return string content from file
       std::string impl = std::string ((std::istreambuf_iterator<char>(bin)), std::istreambuf_iterator<char>());
       return std::make_tuple( kernel_name, impl_location, std::optional(impl) );
      
     }
     else return std::make_tuple(kernel_name, impl_location, std::optional<std::string>{} );
    
  } );

  return binaries;
}

void ocl_runtime::_append_programs_kernels( auto binaries )
{
  //at this point all the programs are open in binaries
  std::ranges::for_each( _contexts, [&](auto& ocl_ctx){
    auto program_flow = binaries | std::views::transform( [&]( auto binary )
    {
      auto kernel_name    = std::get<0>(binary);
      auto impl_location  = std::get<1>(binary);
      auto o_bin          = std::get<2>(binary);
 
      cl_int err;
      auto ret = ocl_program_t{impl_location};

      if( o_bin )
      {
        //replicate implementation points
        size_t num_dev = ocl_ctx._dev_ids.size();
        auto bin       = o_bin.value();

        auto impls = std::vector(num_dev, bin.c_str() );
        //replicate implement sizes
        auto impl_sizes = std::vector<size_t>(num_dev, bin.size() ); 
        //replicate binary_status
        auto binary_stats = std::vector<int>( num_dev );
     
        //create the program here
        auto program = clCreateProgramWithBinary( ocl_ctx._ctx, num_dev, 
                                                  ocl_ctx._dev_ids.data(), 
                                                  impl_sizes.data(), 
                                                  (const unsigned char **) impls.data(), 
                                                  binary_stats.data(), &err);
        

        if( !std::ranges::all_of(binary_stats, unary_equals<int>{CL_SUCCESS} ) || 
            err != CL_SUCCESS  ) 
          return ret;
        else 
        {
          cl_int errr = clBuildProgram(program, num_dev, ocl_ctx._dev_ids.data(), "", NULL, NULL);
          if( errr != CL_SUCCESS )
          { 
            std::cout << "Failed to Build " << ret.impl_location.value() << std::endl;
            return ret;
          }
          else
          {
            std::cout << "Successfully created program for " << ret.impl_location.value() << std::endl;
            ret.program = program;
            //create the kernel
            auto kernel = clCreateKernel( program, kernel_name.c_str(),  &err);
            if( err == CL_SUCCESS)
              ret.kernels[kernel_name] = kernel;
            else
              ret.kernels[kernel_name] = {};
             
            return ret;
          }

        }
      }
      else return ret;

    } ); //end of program_flow

    std::cout << "processing contexts" << std::endl;
    for( auto program : program_flow ) 
      ocl_ctx._programs.emplace_back(program);

  } ); //end of each context
  std::cout << "end of append" << std::endl;

}


std::optional<std::vector<cl_device_id> >
ocl_runtime::_get_devices( )
{
  cl_uint uiNumAllDevs, num_platforms = 0;
  std::vector< std::vector<cl_device_id > > out;
  cl_int ciErrNum =0;

  //get numbers of platforms available
  ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);

  std::vector<cl_platform_id> plat_ids(num_platforms);

  ciErrNum = clGetPlatformIDs (num_platforms, plat_ids.data(), NULL);

  auto _for_fpga_id = [&](cl_platform_id plat_id)
  {
    std::array<char,1024> plat_name;
    ciErrNum = clGetPlatformInfo (plat_id, CL_PLATFORM_NAME, plat_name.size(), plat_name.data(), NULL);
    if( (ciErrNum == CL_SUCCESS) && subsearch(plat_name, "FPGA")  ) return true;
    else return false;
  };

  auto _to_fpga_did = [&](cl_platform_id plat_id)
  {
    // Get the number of GPU devices available to the platform
    clGetDeviceIDs(plat_id, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &uiNumAllDevs);
    //create storage for device list
    std::vector<cl_device_id> _devices(uiNumAllDevs);
    // Create the device list
    clGetDeviceIDs(plat_id, CL_DEVICE_TYPE_ACCELERATOR, uiNumAllDevs, _devices.data(), NULL);
    //return list of lists
    return _devices;
  
  };

  auto check = [](auto val) { std::vector<cl_device_id> i = val; };

  auto device_ids = plat_ids | std::views::filter(_for_fpga_id) | std::views::transform( _to_fpga_did) | std::views::take(1);
  //cant use normal std::copy because it dodesn't except sentinels of different types
  //ranges::copy( std::begin(device_ids), std::end(device_ids), std::back_inserter(out) );
  std::ranges::for_each( device_ids, [&](auto devices){ out.push_back( devices); } );

  if( out.size() ) {
    std::cout << "Found " << out.size() << " platform(s) with " << out[0].size() << " devices" << std::endl;
    return out[0];
  }
  else 
  {
    std::cout << "Could not find any platforms!" << std::endl;
    return {};
  }


}


