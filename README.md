# FlashWrapper
Source :
  export LD_LIBRARY_PATH=/archive-t2/Design/fpga_computing/wip/FlashWrapper:$LD_LIBRARY_PATH
  source /home/tools/altera/setup_scripts/Intel_Setup_19.1-pro_nalla_385a.sh

Compile line: 
  
  g++ main.cc flash_runtime/*.cc opencl_runtime/*.cc -o host.bin -I. -I/home/tools/altera/19.1-pro/hld/host/include/CL/ -L/home/tools/altera/19.1-pro/hld/host/linux64/lib/ -std=c++2a -lOpenCL

Generating an .so

  g++ flash_runtime/*.cc cpu_runtime/*.cc opencl_runtime/*.cc cuda_runtime/*.cc -o libflash_wrapper.so -I. -I/home/tools/altera/19.1-pro/hld/host/include/CL/ -I/usr/local/cuda-11.1/include/ -L/home/tools/altera/19.1-pro/hld/host/linux64/lib/ -shared -fPIC -fvisibility=hidden -std=c++2a -lOpenCL -lcuda -lpthread -ldl

Building main as object

g++ -c cuda_main.cc -I/archive-t2/Design/fpga_computing/wip/FlashWrapper/ -L /archive-t2/Design/fpga_computing/wip/FlashWrapper/ -lflash_wrapper -lcuda -std=c++2a

Linking kernel files and main with cuda :
nvcc -arch=sm_50 cuda_kernels.o cuda_main.o -o host.bin -I/archive-t2/Design/fpga_computing/wip/FlashWrapper/ -L/archive-t2/Design/fpga_computing/wip/FlashWrapper/ -lflash_wrapper

Building CPU unit test
(from cpu_test folder)
g++ *.cc -o host.bin -I. -I../../ -L../../ -lflash_wrapper -std=c++2a -ldl -lpthread -rdynamic

