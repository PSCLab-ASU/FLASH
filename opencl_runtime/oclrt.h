#include <memory>
#include <iostream>
#include <common.h>
#include <vector>


class oclrt_runtime
{

  public:

    static std::shared_ptr<oclrt_runtime> get_runtime();
 
    status register_kernels(  

    status execute( std::string, uint, std::vector<te_variable>, std::vector<te_variable> ); 


  private:

    oclrt_runtime();

    static  std::shared_ptr<oclrt_runtime> _global_ptr; 

};



