#include <memory>
#include <string>
#include <map>
#include <utils/common.h>

#pragma once

class IFlashableRuntime
{
 
  public:

    virtual status register_kernels(const std::vector<kernel_desc> &, 
                                    std::vector<bool>& ) =0;

    virtual status allocate_buffer( te_variable&, bool& ) =0;

    virtual status deallocate_buffer( std::string , bool& ) =0; //dealloc by buffer_id

    virtual status deallocate_buffers( std::string ) =0;        //dealloc by transaction id

    virtual status transfer_buffer( std::string, void *) =0;    //get data by buffer_id

    virtual status execute(runtime_vars, uint num_of_inputs,
                           std::vector<te_variable> kernel_args, 
                           std::vector<size_t> exec_parms) =0;
    virtual status wait( ulong ) =0;
};


template<typename T>
struct FlashableRuntimeMeta
{
  using type = T;

  using IGetMethod = std::shared_ptr<T>(*)();

  void set_creation( IGetMethod method ) { m_GetRuntime = method; }
  void set_description( std::string desc ) { m_Description = desc; }
  std::string get_description () { return m_Description; }

  std::shared_ptr<T> operator()(){ return m_GetRuntime(); };

  IGetMethod  m_GetRuntime;
  std::string m_Description;
};
