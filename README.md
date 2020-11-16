# FlashWrapper
Source :

  source /home/tools/altera/setup_scripts/Intel_Setup_19.1-pro_nalla_385a.sh

Compile line: 
  
  g++ main.cc flash_runtime/*.cc opencl_runtime/*.cc -o host.bin -I. -I/home/tools/altera/19.1-pro/hld/host/include/CL/ -L/home/tools/altera/19.1-pro/hld/host/linux64/lib/ -std=c++2a -lOpenCL

Generating an .so

  g++ flash_runtime/*.cc opencl_runtime/*.cc cuda_runtime/*.cc -o libflash_wrapper.so -I. -I/home/tools/altera/19.1-pro/hld/host/include/CL/ -I/usr/local/cuda-11.1/include/ -L/home/tools/altera/19.1-pro/hld/host/linux64/lib/ -shared -fPIC -fvisibility=hidden -std=c++2a -lOpenCL -lcuda
