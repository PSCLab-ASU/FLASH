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
{
  std::cout << "Ctor'ing flash_cuda...." << std::endl;
  //grab a handle to each devices primary context
  _retain_cuda_contexts();


}

status flash_cuda::wait( ulong  wid)
{
  //get job from pending jobs  
  
 
  return status{};
}

status flash_cuda::execute(runtime_vars rt_vars, uint num_of_inputs, 
                              std::vector<te_variable> kernel_args, std::vector<size_t> exec_parms)
{
  std::cout << "Executing from cuda-flash_runtime..." << __func__ << std::endl;
  std::cout << "Executing : " << rt_vars.get_lookup() <<" ..."<< std::endl;







  return status{-1};
}

status flash_cuda::register_kernels( const std::vector<kernel_desc>& kds ) 
{
  
  std::cout << "calling flash_cuda::" << __func__<<  std::endl;
  for( auto kernel : kds )
  {
    
    std::cout << "kernel_name = " << kernel._kernel_name << std::endl;
    std::cout << "kernel_location = " << kernel._kernel_definition.value() << std::endl;
    if( kernel._kernel_type == kernel_t::INT_BIN ||
        kernel._kernel_type == kernel_t::EXT_BIN )
      _process_external_binary( kernel );
    else
      std::cout << "registration format not supported" << std::endl;
  }

  //reset back to initial context
  _reset_cuda_ctx();

  return {}; 
}


template<MEM_MOVE dir>
status pending_job_t::transfer(uint src, uint dst)
{

  if( dir == MEM_MOVE::TO_DEVICE ) 
  {
    //copy data to device buffer
  }
  else
  {
    //copy data to device buffer

  }

  return status{};

}

//does not support READ_WRITE parameters yet
template<MEM_MOVE dir>
status pending_job_t::transfer_all()
{

  if( dir == MEM_MOVE::TO_DEVICE ) 
  {
    for( auto i : std::views::iota((uint)0, num_inputs) ) 
      transfer<dir>(i, i);
  }
  else
  {
    for( auto i : std::views::iota(num_inputs, kernel_args.size() ) ) 
      transfer( i, i );

  }
  return status{};
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
  std::string func_regex = "[^\\. ]+" + func_name + "[^\\.]";
  std::regex fregex (func_regex);
  for(size_t i = 1; i < mod_ind.size()-1; i++)
  {
    std::string subst = cdata.substr(mod_ind[i], mod_ind[i+1] - mod_ind[i] );
   
    if( std::regex_search(subst, sof_match, fregex) ) 
    {
      for (size_t j = 0; j < sof_match.size(); ++j) 
      {
        int ret = cuModuleLoadFatBinary(&cuModule, &((unsigned char *) nvFCubinBase)[mod_ind[i]]  );    
     
        if( ret == CUDA_SUCCESS ) 
        {
          ret = cuModuleGetFunction(&khw, cuModule, sof_match[j].str().data() );

	  if( ret == CUDA_SUCCESS )
	  {
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
	    //add function
	    _kernel_comps.push_back(
	  		    cuda_function{
			      function_name, mangled_name, khw
			    }
			  );
	    //create kernel_library entry
	    _kernel_lib.push_back( 
			     cuda_kernel{ function_name, mod.get_id(), ctx.get_id(), offset }
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

      bool isElf = elf_func( elf_is_elf64, file);

      if( isElf )
      {
        bool contains_fatbinary = elf_func(elf64_get_section_header_by_name, 
			                   file, (const Elf64_Ehdr *) &elf_header, nvFatBinHdr.c_str(), &header);
	if( contains_fatbinary ) 
	{
          //Add the modules to the list
          _kernel_comps.push_back
	  ( 
            cuda_module{ location, file, file_size, binary_mmap_ptr, 
	                 fd, header.sh_addr, header.sh_size }
	  );
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
        _kernel_comps.push_back
	( 
          cuda_context{ std::to_string( random_number() ), 
			device, 
			ctx } 
	);
      }
    }
  }
}
