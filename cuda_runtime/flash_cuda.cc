#include "cuda_runtime/flash_cuda.h"
#include <iostream>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <climits>
#include <sys/mman.h>
#include <sys/stat.h> 
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <regex>
#include <vector>

/* Registers the factory with flash factory*/
bool flash_cuda::_registered = FlashableRuntimeFactory::Register(
                                flash_cuda::get_factory_name(),
                                flash_cuda::get_runtime() );

std::shared_ptr<flash_cuda> flash_cuda::_global_ptr;


FlashableRuntimeMeta<IFlashableRuntime> flash_cuda::get_runtime()
{
  //automatic polymorphism to base classa
  FlashableRuntimeMeta<IFlashableRuntime> out{ (std::shared_ptr<IFlashableRuntime> (*)()) get_singleton, 
                                                get_factory_desc() };

  return out;
}


std::shared_ptr<flash_cuda> flash_cuda::get_singleton()
{

  if( _global_ptr ) return _global_ptr;
  else return _global_ptr = std::shared_ptr<flash_cuda>( new flash_cuda() );

}

flash_cuda::flash_cuda()
: _kernel_lib{ _kernel_comps }
{
  std::cout << "Ctor'ing flash_cuda...." << std::endl;
  //initialize driver API
  cuInit(0);
  //grab a handle to each devices primary context
  _retain_cuda_contexts();


}

status flash_cuda::register_kernels( const std::vector<kernel_desc>& kds ) 
{
  
  std::cout << "calling flash_cuda::" << __func__<<  std::endl;
  for( auto kernel : kds )
  {
    
    if( kernel._kernel_type == kernel_t::INT_BIN ||
        kernel._kernel_type == kernel_t::EXT_BIN )
    {
      std::cout << "registering " << kernel._kernel_name
	        <<" in "<< kernel._kernel_definition.value() << std::endl;
      _process_external_binary( kernel );
    }
    else
      std::cout << "registration format not supported" << std::endl;
  }

  //reset back to initial context
  _reset_cuda_ctx();

  return {}; 
}

status flash_cuda::wait( ulong  wid)
{
  std::cout << "Enttering " << __func__ << std::endl;
  //get job from pending jobs  
  auto entry = _pending_jobs.find( wid);

  if( entry != _pending_jobs.end() ) 
  {
    //set context to the job related to wid;
    auto pjob = entry->second;
    _kernel_comps.set_active_context( pjob.cuCtx_key );
    //blocks until job is complete
    std::cout << "Sync'ing ctx..." << std::endl;
    int ret = cuCtxSynchronize();
    std::cout << "waiting complete" << std::endl;
    //move data from device buffer to host buffera
    std::vector<int> rets;
    int i=0;
    std::ranges::transform(pjob.kernel_args, pjob.device_buffers, std::back_inserter( rets ),
		           [&](auto h_args, auto d_args)
		           {
			     //only move output buffers
			     if( i++; i > pjob.nInputs )
			     {
			       std::cout << "transfering buffer " << i << " from device to host" << std::endl;
                               return cuMemcpyDtoH ( h_args.get_data(), d_args, h_args.get_bytes() );
			     }
			     else return CUDA_SUCCESS;
  
		           } );

    bool success = std::ranges::all_of( rets, unary_equals{(int)CUDA_SUCCESS} );

    if( !success ) std::cout << "Could not move all device buffers to host" << std::endl;

    //remove job
    _pending_jobs.erase( wid);
  }
  else std::cout << "Could not find pending job..." << std::endl;
 
  return status{};
}

status flash_cuda::execute(runtime_vars rt_vars, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms)
{
  std::cout << "Executing from cuda-flash_runtime..." << __func__ << std::endl;
  std::cout << "Executing : " << rt_vars.get_lookup() <<" ..."<< std::endl;

  //set current context to context with the least amount of work
  _set_least_active_gpu_ctx();
  std::string kernel_name = rt_vars.kernel_name_override?rt_vars.kernel_name_override.value():rt_vars.get_lookup();
  auto func_binary = _kernel_lib.get_cufunction_for_current_ctx( kernel_name, rt_vars.kernel_impl_override );
  auto ctx_key     = _kernel_comps.get_current_ctx_key();
  //auto input_kargs = std::vector<te_variable>( kernel_args.begin(), kernel_args.begin() + num_of_inputs); 

  if( func_binary )
  {
    int i =0;
    std::cout << "Found executable kernel..." << std::endl;
    std::vector<CUdeviceptr> device_buffers( kernel_args.size() );
    //std::ranges::transform( kernel_args, std::back_inserter( device_buffers ), 
    std::ranges::transform( kernel_args, device_buffers.begin(), 
                            [&](auto host_buffer )
			    {
			      CUdeviceptr dev_ptr;
			      size_t buffer_size = host_buffer.get_bytes();
			      std::cout << "Allocating device memory : " << buffer_size << std::endl;
                              auto ret = cuMemAlloc(&dev_ptr, buffer_size );

			      if( ret == CUDA_SUCCESS )
			      {
			        //only transfer the inputs from H -> D
			        if( i++; i <= num_of_inputs )
			        {
			          std::cout << "Transfering buffer from host to device : " << i << std::endl;
			          ret = cuMemcpyHtoD( dev_ptr, host_buffer.get_data(), buffer_size );
				  if( ret != CUDA_SUCCESS)
				    std::cout << "Failed to send data to device" << std::endl;
			        }
			      }
			      else std::cout << "Failed to allocate device memory" << std::endl;

			      return dev_ptr;
			    });	

    //create pending job entry
    //WARNING MAKE SURE this void * is valid during the invocation of the kernel
    //void * is a CUDA back to erase the type of the device pointer.
    std::vector<void *> void_devs_buffs;
    std::ranges::transform(device_buffers, std::back_inserter( void_devs_buffs ),
    		           [](auto& device_buffer) { return (void *) &device_buffer; } );

    //launch kernel
    std::vector<size_t> dims(8, 1);
    dims[7] = dims[6] = 0;
    std::ranges::copy(exec_parms, dims.begin() );

    std::ranges::copy( dims, std::ostream_iterator<size_t>{std::cout, ", "} );
    std::cout << std::endl;
    int ret = cuLaunchKernel(func_binary.value(), dims[0], dims[1], dims[2], dims[3], 
                             dims[4], dims[5], dims[6], nullptr, void_devs_buffs.data(), 0);
    //int ret = cuLaunchKernel(func_binary.value(), 1, 1, 1, 1, 1, 1, 0, 0, void_devs_buffs.data(), 0);

    if( ret == CUDA_SUCCESS)
    {
      std::cout << "Kernel successfully launched..." << std::endl;
      ulong wid = random_number(); 
      //pending jobs

      _pending_jobs.emplace(std::piecewise_construct,
		            std::forward_as_tuple(wid),
			    std::forward_as_tuple(ctx_key, kernel_args, device_buffers, num_of_inputs ) );

      _job_count_per_ctx.at(ctx_key) += 1;

      auto stat = status{0, wid };

      return stat;
    }
    else std::cout << "Could not launch " << rt_vars.get_lookup() << " : " << ret << std::endl;
    
  }
  else
  {
    std::cout << "Could not find function to execute" << std::endl;
  }

  return status{-1};
}


void flash_cuda::_reset_cuda_ctx()
{
  _kernel_comps.contexts[0].set_context_to_current();
}

int flash_cuda::_find_cubin_offset(void * nvFCubinBase, size_t fbin_len, std::string func_name, size_t * cubin_offset, std::string * mangled_name)
{
  std::smatch sof_match;
  CUmodule cuModule;
  CUfunction khw;    
  size_t offset =0;
  auto cdata = std::string( (char *) nvFCubinBase, fbin_len);
  std::vector<size_t> mod_ind = {0, cdata.size() };

  std::regex sof_regex ("\x50\xed\x55\xba\x01\x00\x10\x00");
  while(std::regex_search(cdata, sof_match, sof_regex)) 
  {
    for (size_t i = 0; i < sof_match.size(); ++i) 
    {
      auto abs_indx = sof_match.position(i) + offset;
      offset += sof_match.position(i) + 5;
      mod_ind.insert(std::next(mod_ind.begin(), mod_ind.size() - 1 ), abs_indx );
    }    
   
    cdata = sof_match.suffix(); 
  }

  //find function
  std::string func_regex = "[^\\. ]+" + func_name + "[^\\.]+";
  std::regex fregex (func_regex);

  for(size_t i = 1; i < mod_ind.size()-1; i++)
  {
    std::string subst = cdata.substr(mod_ind[i], mod_ind[i+1] - mod_ind[i] );

    int ret = cuModuleLoadFatBinary(&cuModule, &((unsigned char *) nvFCubinBase)[mod_ind[i]]  );    

    if( ret == CUDA_SUCCESS ) 
    {
      if( std::regex_search(subst, sof_match, fregex) ) 
      {
        for (size_t j = 0; j < sof_match.size(); ++j) 
        {
          //std::cout << "Checking function : " << sof_match[j].str().data() << std::endl;  
          ret = cuModuleGetFunction(&khw, cuModule, sof_match[j].str().data() );

	  if( ret == CUDA_SUCCESS )
	  {
            //std::cout << "Successfuly found cubin for ..." <<  sof_match[j] <<  std::endl;
            *mangled_name  = sof_match[j];
	    *cubin_offset =  mod_ind[i];
            cuModuleUnload(cuModule);
            return 0;
	  } //end of loading function
	  
          cuModuleUnload(cuModule);
        } //end of loading module
      } //end of function mapping    
    } //end of regex_search
  } //end of for loop index
  
  return -1;
}	


void flash_cuda::_process_external_binary( const kernel_desc& kd)
{
  //adding the module	
  _add_cuda_modules( kd._kernel_definition.value() );
  // add function to the library
  _add_cuda_function( kd._kernel_name );
}

void flash_cuda::_add_cuda_function( std::string function_name )
{
  std::cout << "entering " << __func__ << std::endl;

  //add a function and the module offset
  for(cuda_module& mod : _kernel_comps.modules )
  {
    size_t offset;
    std::string mangled_name;

    //find a valid cubin file within the fatcubin section
    unsigned char * data = mod.get_fbin_data<unsigned char>();      

    for( cuda_context& ctx : _kernel_comps.contexts)
    {
      //set current thread to context
      ctx.set_context_to_current();


      int err = _find_cubin_offset( data, mod.get_fbin_size(), 
			            function_name, &offset, &mangled_name);
      if(err == 0)
      {
	//create and add module to list
        bool added = mod.test_and_add( offset );

	//module is added
	if( added ) 
	{
	  CUfunction khw;
          err = cuModuleGetFunction(&khw, *mod.get_module(offset), mangled_name.c_str());
	  if( err == CUDA_SUCCESS)
	  {
            std::cout << "Found function entry point for : " << function_name << " : " << mangled_name <<std::endl;
	    std::string func_id = std::to_string( random_number() );
	    //add function
	    _kernel_comps.push_back(
	  		    cuda_function{
			      func_id,
			      function_name, mangled_name, khw
			    }
			  );
	    //create kernel_library entry
	    _kernel_lib.push_back( 
			     cuda_kernel{ func_id, function_name, mod.get_id(), ctx.get_id(), offset }
			    );
            
	  } //added function to functions
	} //add module to modules
      } //finding the cubin     
    } //context for loop
  } //module for loop

} 


void flash_cuda::_add_cuda_modules( std::string location )
{
  void * binary_mmap_ptr;
  Elf64_Ehdr elf_header;
  Elf64_Shdr header;
  std::string nvFatBinHdr = ".nv_fatbin";

  struct stat sb;
  size_t file_size=0;

  if( _kernel_comps.module_exists( location ) )
  {
    std::cout << "module already exists " << std::endl; 
    return; 
  }

  FILE *file = fopen( location.c_str(), "rb" );
  int fd     = fileno( file );
  
  auto elf_func = [&](auto func, auto ... parms)->bool {
    bool b = func( parms ...);
    fseek(file, 0, SEEK_SET);
    return b;
  };

  //check for a valid file descriptor and file pointer
  if( (fd >= 0) && file != nullptr )
  {
    if( fstat(fd, &sb ) != -1 )
    {
      file_size = sb.st_size;
      //mmap binaryt
      binary_mmap_ptr = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);

      if( binary_mmap_ptr != nullptr )
      {
        bool isElf = elf_func( elf_is_elf64, file);

        if( isElf )
        {
	  //get elf header
          elf_func( elf64_get_elf_header, file, &elf_header);

          bool contains_fatbinary = elf_func(elf64_get_section_header_by_name, 
	    		                     file, (const Elf64_Ehdr *) &elf_header, nvFatBinHdr.c_str(), &header);
	  if( contains_fatbinary ) 
	  {
            //Add the modules to the list
            _kernel_comps.push_back
	    ( 
              cuda_module{ location, file, file_size, binary_mmap_ptr, 
	                   fd, header.sh_offset, header.sh_size }
	    );
	  }
        }
      }
    }
  }
}

void flash_cuda::_retain_cuda_contexts()
{
  CUcontext ctx;
  CUdevice device;
  CUresult result;

  int deviceCount = 0;
  
  //get the number od devices
  cuDeviceGetCount( &deviceCount );  

  std::cout << "Found " << deviceCount << " devices" << std::endl;
  //get current context, and device
  for (int i : std::views::iota(0, deviceCount) )
  {
    //get device handles
    result = cuDeviceGet (&device, i);

    if( result == CUDA_SUCCESS)
    {
      //get contexts
      result = cuDevicePrimaryCtxRetain ( &ctx, device );

      if( result == CUDA_SUCCESS)
      {
	//push to compoements.context
        auto ctx_key = std::to_string( random_number() );
        _kernel_comps.push_back
	( 
          cuda_context{ ctx_key, 
			device, 
			ctx } 
	);

        //iontializing jobs table
	_job_count_per_ctx.insert({ctx_key, 0});

      }
    }
  }
}

void flash_cuda::_set_least_active_gpu_ctx()
{
  auto lowest = std::ranges::min_element(_job_count_per_ctx, 
  		                          {}, &std::map<std::string,ulong>::value_type::second);
  if( lowest != _job_count_per_ctx.end() )
    _kernel_comps.set_active_context( lowest->first );
  else std::cout << "Could not find gpu_ctx entry" << std::endl;
}

