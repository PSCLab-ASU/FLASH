#include "opencl_runtime/oclrt.h"
#include <iostream>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <climits>



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
  if( device_ids ) 
  {
    //save off device list
    //
    for( auto did : *device_ids) _device_usage_table.emplace(did, 0);  

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
  
      if( err == CL_SUCCESS ) 
      { 
        auto octx = this->_contexts.emplace_back( ctx, device_ss );
  
        //create device buffers
        std::ranges::for_each(device_ss, [&](auto device_id)
        {
          auto queue = clCreateCommandQueueWithProperties( ctx, device_id, NULL, &err);
          
          if( err == CL_SUCCESS ) 
          { 
            octx._queues.emplace_back( queue );
          } 
          else std::cout << "Could not create command queues" << std::endl;
  
        } );
      }
      else std::cout << "Could not create context for device(s)" << std::endl; 
  
    } );
  }
  else std::cout << "No devices found " << std::endl;
}

status ocl_runtime::wait( ulong  wid)
{
  //get job from pending jobs  
  if(auto pjob = _pending_jobs.find(wid); 
     pjob != _pending_jobs.end() )
  {
    auto transfer_status = pjob->second.transfer_all(); 
  }
  else std::cout << "Could not find wid = " << wid << std::endl;

  return status{};
}

status ocl_runtime::execute(runtime_vars rt_vars, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms)
{
  std::cout << "Executing from opencl_runtime..." << rt_vars.get_lookup() << std::endl;
  uint n_inputs = num_of_inputs;
  uint arg_index=0;
  cl_event c_event;
  ulong wid;

  auto [ctx, kernel, queue, valid, dev_id] = _try_get_exec_parms( rt_vars.get_lookup() );

  //converts host buffers to cl_mem object and transfer data to device
  auto hbuff_to_clmem = [&]( auto hbuffer ) -> cl_mem
  {
    int buffer_properties;
    cl_int err;
    size_t sz = hbuffer.type_size * hbuffer.vec_size;

    if( n_inputs-- ) buffer_properties = CL_MEM_READ_ONLY;
    else buffer_properties = CL_MEM_READ_WRITE;   
 
    auto dbuffer = clCreateBuffer( ctx, buffer_properties, sz, NULL, &err);
    
    if( err != CL_SUCCESS) throw std::runtime_error("Could not create buffer");
   

    return dbuffer;
  };

  //sets cl_mem object to corresponding argument
  auto set_kernel_args = [&] (auto dbuffer) -> std::pair<cl_mem, int>
  {
     
     return std::make_pair( dbuffer, 
                            (int) clSetKernelArg(kernel, arg_index++, sizeof(cl_mem), (void *) dbuffer  ) ); 
  };
  

  if( valid )
  {
    //transform host buffers to cl_mem buffers
    auto buffer_transfers = kernel_args | std::views::transform( hbuff_to_clmem ) | 
                            std::views::transform( set_kernel_args );


    bool complete = false;
    std::vector<cl_mem> dev_buffers;

    for(auto [buffer, cl_status] : buffer_transfers)
    {
      complete &= (cl_status == CL_SUCCESS);
      dev_buffers.push_back(buffer);  
    }
   
    //move data to device 
    wid = _add_to_pending_jobs( num_of_inputs, kernel_args, dev_buffers, queue );
    auto pentry = _pending_jobs.at(wid);

    //move all inputs to device includeing READ_WRITES
    //defaults to HOST
    pentry.transfer_all<MEM_MOVE::TO_DEVICE>();
    
    if( complete )
    { 
      int ciErrNum;
      //execute method input args are ready
      ciErrNum = clEnqueueNDRangeKernel(queue, kernel, exec_parms.size(), NULL, 
                                        exec_parms.data(), NULL, 0, NULL, &c_event);      
      //increment device_Id
      //useful when I add out of order execution for load balancing
      _device_usage_table.at( dev_id )++; 

      //remove this line after transactio nsupport
      pentry.set_event( std::move(c_event) );

      auto stat = status{0, wid };

      return stat;

      //_process_output( num_of_inputs, kernel_args, dev_buffers );

    } else std::cout << "Could not transfer buffers to device" << std::endl;
    
  } else std::cout << "Could not find kernel" << std::endl;
 

  return status{-1};
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

std::tuple<cl_context, cl_kernel, cl_command_queue, bool, cl_device_id>
ocl_runtime::_try_get_exec_parms( std::string method_name)
{
   auto single_ctx_rule = [](auto ctx ) { return ctx._dev_ids.size() == 1; };
   uint dev_occupancy = UINT_MAX;
   cl_context ctx; cl_kernel clk; cl_command_queue cq; bool valid;
   cl_device_id dev_id;

   //Crude available device lookup. Only works for single device
   auto single_ctx = _contexts | std::views::filter(single_ctx_rule);

   std::ranges::for_each( single_ctx, [&](auto octx) 
   {
     std::ranges::for_each( octx._programs, [&](auto oprog)
     {
       for(auto[k_name, k_obj] : oprog.kernels )
       {
         if ( (k_name == method_name) && k_obj)
           if( _device_usage_table.at( octx._dev_ids.at(0) ) < dev_occupancy)
           {
             ctx    = octx._ctx;
             clk    = k_obj.value();
             cq     = octx._queues[0];
             dev_id = octx._dev_ids.at(0);       
             dev_occupancy = _device_usage_table.at( dev_id );
             valid = true;
           }
         
       }
     });
   });
   
   return std::make_tuple(ctx, clk, cq, valid, dev_id);
}

ulong ocl_runtime::_add_to_pending_jobs( uint num_input, 
                                         std::vector<te_variable> kernel_args, 
                                         std::vector<cl_mem> device_buffer,
                                         cl_command_queue cq)
{
  ulong wid = random_number();

  auto pj = pending_job_t{ .num_inputs = num_input,.kernel_args = kernel_args,.device_buffers = device_buffer,.cq = cq  };
  _pending_jobs[wid] = std::move(pj);

  return wid;
}

template<MEM_MOVE dir>
status pending_job_t::transfer(uint src, uint dst)
{
  cl_int err;
  if( kernel_args.size() != device_buffers.size())
    std::cout << "Error mismatch pending job sizes" << std::endl;

  if( dir == MEM_MOVE::TO_DEVICE ) 
  {
    //copy data to device buffer
    std::cout << "Moving arg "<< dst << " to device " << std::endl;
    size_t sz = kernel_args[src].type_size * kernel_args[src].vec_size;
    err = clEnqueueWriteBuffer(cq, device_buffers[dst], CL_TRUE, 0, sz, kernel_args[src].data, 0, NULL, NULL);

    if( err != CL_SUCCESS) throw std::runtime_error("Could not transfer buffer from host to device");
  }
  else
  {
    //copy data to device buffer
    std::cout << "Moving device bufer "<< dst << " to host " << std::endl;
    size_t sz = kernel_args[dst].type_size * kernel_args[dst].vec_size;
    err = clEnqueueReadBuffer(cq, device_buffers[src], CL_TRUE, 0, sz, kernel_args[dst].data, 1, &event, NULL);

    if( err != CL_SUCCESS) throw std::runtime_error("Could not transfer buffer from host to device");

  }

  return status{};

}

//does not support READ_WRITE parameters yet
template<MEM_MOVE dir>
status pending_job_t::transfer_all()
{

  if( dir == MEM_MOVE::TO_DEVICE ) 
  {
    for( auto i : std::views::iota(num_inputs) ) 
      transfer<dir>(i, i);
  }
  else
  {
    for( auto i : std::views::iota(num_inputs, kernel_args.size() ) ) 
      transfer( i, i );

  }
  return status{};
}

