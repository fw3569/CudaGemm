#### 介绍
一个CUDA GEMM算子

#### 如何构建
构建依赖`nvcc`和`msvc`和`Ninja`  
请自行更改或在命令行指定`CMAKE_CUDA_HOST_COMPILER`，
```
cmake -G "Ninja Multi-Config" -S ./ -B build
cmake --build ./build --config RelWithDebInfo
```

#### 技术特性
分块优化：结合Shared Memory与Register Blocking分块，减少Global Memory访存，提升FFMA计算密度。  
访存对齐：使用alignas(128)确保首地址对齐缓存行，支持float4向量化指令和合并访存，减少内存事务分裂。  
线性地址映射：通过连续的线程-数据索引设计，使访存波前与物理Banks天然契合，原生消除Shared Memory Bank Conflict。  
硬件资源约束：使用launch_bounds控制寄存器与共享内存用量，确保维持活跃Warp数。


#### 性能测试

| 指标 | 参数 |
| :--- | :--- |
| **GPU** | NVIDIA GeForce MX450 |
| **Compute Capability** | 7.5 (Turing) |
| **SM 数量** | 14 |
| **显存带宽** | 80 GB/s |
| **理论算力** | 2.5 TFLOPS |

在`mx450`上测试$1024^3$的GEMM计算性能达到`cublasSgemm`的约`85%`，用`RelWithDebInfo`编译

```
kernel time : 1.21598ms, total time: 4.41777ms, tflops: 1.76606
cublas kernel time: 1.05059ms, rate: 0.86399
```

下图是python下调用dll测试benchmark的结果,用`-O3`和`/O2`编译

![benchmark results](./benchmark/benchmark_results.png)

下面是nsight的测试结果

![GPU Speed Of Light Throughput](./document/img/屏幕截图%202026-02-25%20141732.png)

![Roofline](./document/img/屏幕截图%202026-02-25%20133524.png)

位于Roofline模型右侧处于计算瓶颈，达到74.86%的计算吞吐量

![Occupancy](./document/img/屏幕截图%202026-02-25%20142218.png)

![Impact of Varying Register Count Per Thread](./document/img/屏幕截图%202026-02-25%20133958.png)

![Impact of Varying Shared Memory Usage Per Block](./document/img/屏幕截图%202026-02-25%20134018.png)

__launch_bounds__限制后，寄存器和共享内存共同限制了Occupancy，并行warp数量达到理论值的90%，有效掩盖访存延迟

![Warp State](./document/img/屏幕截图%202026-02-25%20141241.png)

![Instruction Stall](./document/img/屏幕截图%202026-02-25%20142736.png)

Math Pipe Throttle达到31.20%，证明运算单元处于满载排队状态
