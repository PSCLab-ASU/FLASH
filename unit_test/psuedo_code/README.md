The purpose of this section is to demonstrate typical use cases and pattern taking a classical approach from known methods from existing work
versus how FLASH implements the corresponding use case. Each files includes puesdo/real code in which application need to implement to 
enable application with corresponding functionality. These use cases are base-effort implementaitons for achieving the functionality.
Some of these functionality are still a work in progress in FLASH, so psuedo code is provided in this scenario. 

1. Deep Pipeline (deep_pipeline.ex):

   - Classically comparing pipelines, versus creating micro graphs for lazily executing and optimizing pipelines with FLASH.

   - Note: FLASH will provide small graph optimizations in the future since it constructs and executes a pipeline lazily.

2. Multi-homogenous Accelerator Use Case (mutl_homo_accel.ex):

   - Dispatching similar kernels across multi homogenous accelerators known methods, vs. FLASH.

   - Note: FLASH automatically load balances across all accelerators available implicitly on submition boundaries.

3. Multi-heterogenous Accelerator Use Case (multi_heter_accel.ex):

   - Comparing known approaches for dispatching similar kernels  kernels across various accelerators of different architectures, known methods vs. FLASH.

   - Note: FLASH automatically load balances across all accelerators available implicitly on submition boundaries.

4. Multi-implementation, Single Accelerator Use Case (multi_impl.ex):

   - Comparing known approaches for dispatching multiple kernels across similar architectures, known methods vs. FLASH.

   - Note: Since FLASH is a singleton pattern, implementations are cached and can dispatched implementation throughout the life of the application 

5. Multi-implementation, Multi-Accelerator Use Case (multi_impl.ex):

   - Comparing known approaches for dispatching multiple kernels across various accelerators of different architectures, known methods vs. FLASH.

   - Note: FLASH can maintain multiple implementations with the same lookup value. Similar to virtual dispatching.

6. Implicit Barrier Use Case (implicit_barrier.ex):

   - Comparing known approaches for invoking barriers vs. FLASH barriers.

   - Note: FLASH disconnects the barriers from the implementation of the kernels and instructs the runtime.

7. Multi-Partition, Multi-Accelerator Use Case ( multi_part.ex ):

   - Comparing known approaches for partitioning a single kernel launch across multiple accelerators. 

   - Note: The FLASH API maintains a kernel description that can have a partition function to decompose single kernels launches to multiple kernel launches.
           This enable greater parallelism driven by the application
