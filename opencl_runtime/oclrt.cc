#include "opencl_runtime/oclrt.h"
#include <iostream>
#include <ranges>
#include <algorithm>
#include <fstream>

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
    auto device_ss = std::vector(std::begin(dev_set), std::end(dev_set));

    //create contexts 
    auto ctx = clCreateContext( NULL, device_ss.size(), device_ss.data(), NULL, NULL, &err);
    this->_contexts.emplace_back( ctx, device_ss );

  } );

}

status ocl_runtime::execute(std::string kernel_name, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<te_variable> exec_parms)
{
  std::cout << "Executing from opencl_runtime..." << std::endl;
  return {};
}

status ocl_runtime::register_kernels( const std::vector<kernel_desc>& kds ) 
{
  
  std::cout << "calling ocl_runtime::" << __func__<<  std::endl;
  auto kern = kernel_t::EXT_BIN;
  
  if( std::ranges::all_of(kds, unary_equals{kern}, &kernel_desc::_kernel_type) )
  {

    auto binaries = _read_kernel_files(kds);

    //really nice notation
    //auto program_flow = _context & all(binaries) | std::views::transform( [](auto ctx, auto binary){} )
    
    _append_programs( binaries );   

  }
  else
  {
    std::cout << "Invalid input args..." << std::endl;
  }

  return {}; 
}

std::vector<std::pair<std::string, std::optional<std::string> > >
ocl_runtime::_read_kernel_files( const std::vector<kernel_desc>& kds)
{
  std::vector<std::pair<std::string, std::optional<std::string> > > binaries;
  //read alll file locations
  std::ranges::transform( kds, std::back_inserter(binaries), 
  [](auto kd)
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
       return std::make_pair(kernel_name, std::optional<std::string>{} );
      }
      //return string content from file
      std::string impl = std::string ((std::istreambuf_iterator<char>(bin)), std::istreambuf_iterator<char>());
      return std::make_pair( kernel_name, std::optional(impl) );
    }
    else return std::make_pair(kernel_name, std::optional<std::string>{} );
    
  } );

  return binaries;
}

void ocl_runtime::_append_programs( auto binaries )
{
  //at this point all the programs are open in binaries
  std::ranges::for_each( _contexts, [&](auto& ocl_ctx){
    auto program_flow = binaries | std::views::transform( [&]( auto binary )
    {
      cl_int err;
      auto ret = std::pair<std::string, std::optional<cl_program> >(binary.first, {} );

      if( binary.second )
      {
        //replicate implementation points
        size_t num_dev = ocl_ctx._dev_ids.size();
        auto bin       = binary.second.value();

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
            err != CL_SUCCESS ) 
          return ret;
        else 
        {
          std::cout << "Successfully created program for " << ret.first << std::endl;
          return ret = std::make_pair(binary.first, program);
        }
      }
      else return ret;

    } ); //end of program_flow

    for( auto binary : program_flow ) 
      ocl_ctx._programs.emplace(binary.first, binary.second);

  } ); //end of each context

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


