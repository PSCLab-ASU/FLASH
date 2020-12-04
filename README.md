# FlashWrapper Introduction

The FlashWrapper is a framework that fascilitates the interoperability and portability of application towards hetereogenous accelerators. It decouples the application and allow dynamic dispatch of kernels on different accelerators while minimizing the refactoring efforts of the applications for current and future accelerators. 

# Building shared object (.so) for default (CPU) Backends

1. make OR make all

The output .so will be placed in : ./build/lib64 and headers will be in ./build/include

# Building shared object with specific backends
valid backends are : cpu_runtime, cuda_runtime, opencl_runtime (Intel FPGA-only)

1. make FLASH_VARIANT=[backend[,...]]            ex. make FLASH_VARIANT=cpu_runtime,cuda_runtime #enables CPU and CUDA backends 


# Test builds notes

Building applications require the use of c++20 features, however NVCC doesn't support it yet. The object file(s) that contains Flash logic must be compiled into objects via a C++ 20 enabled compiler. The kernels need to be compiled with NVCC. 

Ex.
  Building main with C++20
    g++ -c cuda_main.cc -o cuda_main.o -I./build/include -L./build/lib64 -lflash_wrapper -lcuda -std=c++2a
  
  Building CUDA kernels
    nvcc -arch=sm_50 -c cuda_kernels.cu -o cuda_kernels.o

  Linking main with cuda object files:
    nvcc -arch=sm_50 cuda_kernels.o cuda_main.o -o host.bin -I./build/include -L./build/lib64 -lflash_wrapper -lcuda

## Building CPU unit test

CPU kernels in the form of free functions or member functions must be built with -rdynamic, -ldl and -lpthread

Member functions must be  attributed with [[gnu::used]] if thier only invokation is via the flash runtime else disregard the attribute. Free functions do not require the function attribute.

Ex. of member function attribute 

```  
  struct TEST 
  {
    [[gnu::used]]
    void hello_world(){}    
    int i=0;
  };
```

### The CPU runtime engine uses a single method for indicating which work items is currently being executed. It is an Nth dimensional indexing system driven by the "defer" or "exec" interface.

  Ex. defer(dim1, dim2, dim3,..., dimN) or exec(dim1, dim2, dim3,..., dimN)  #dims[N] are of type size_t
    
    defer( 3, 3, 3) with N=3 creates 27 total work items
    1. {0, 0, 0}  10. {0, 0, 1}  19. {0, 0, 2}
    2. {1, 0, 0}  11. {1, 0, 1}  20. {1, 0, 2}
    3. {2, 0, 0}  12. {2, 0, 1}  21. {2, 0, 2}
    4. {0, 1, 0}  13. {0, 1, 1}  22. {0, 1, 2}
    5. {1, 1, 0}  14. {1, 1, 1}  23. {1, 1, 2}
    6. {2, 1, 0}  15. {2, 1, 1}  24. {2, 1, 2}
    7. {0, 2, 0}  16. {0, 2, 1}  25. {0, 2, 2}
    8. {1, 2, 0}  17. {1, 2, 1}  26. {1, 2, 2}
    9. {2, 2, 0}  18. {2, 2, 1}  27. {2, 2, 2}
    
The kernels can make a call to size_t get_indices( int dim) to retrieve work item information.
```
Ex. 
  void elementwise_matrix_multiplication( float * a, float * b, float * c)
  {
    auto x = get_indices(0); #Get Dimension 0 current indices 
    c[x] = a[x] * b[x];
    return;
  }
```
## Example compilation

  Ex g++ main.cc -o host.bin -I./build/include -lflash_wrapper -std=c++2a -ldl -lpthread -rdynamic

  *Remember to point LD_LIBRARY_PATH to [dir]/build/lib64
