The purpose of this section is to demonstrate typical use cases and pattern taking a classical approach from known methods from existing work
versus how FLASH implements the corresponding use case. Each files includes puesdo/real code in which application need to implement to 
enable application with corresponding functionality. These use cases are base-effort implementaitons for achieving the functionality.
Some of these functionality are still a work in progress in FLASH, so psuedo code is provided in this scenario. 

Deep Pipeline (deep_pipeline.ex):

  Classically comparing pipelines, versus creating micro graphs for lazily executing and optimizing pipelines with FLASH.

Multi-homogenous Accelerator Use Case (mutl_homo_accel.ex):

  Dispatching similar kernels across multi homogenous accelerators known methods, vs. FLASH.

Multi-heterogenous Accelerator Use Case (multi_heter_accel.ex):

  Comparing known approaches for dispatching similar kernels  kernels across various accelerators of different architectures, known methods vs. FLASH.

Multi-implementation, Single Accelerator Use Case (multi_impl.ex):

  Comparing known approaches for dispatching multiple kernels across similar architectures, known methods vs. FLASH.

Multi-implementation, Multi-Accelerator Use Case (multi_impl.ex):

  Comparing known approaches for dispatching multiple kernels across various accelerators of different architectures, known methods vs. FLASH.

Implicit Barrier Use Case (implicit_barrier.ex):

  Comparing known approaches for invoking barriers vs. FLASH barriers.

Multi-Partition, Multi-Accelerator Use Case ( multi_part.ex ):

  Comparing known approaches for partitioning a single kernel launch across multiple accelerators. 
