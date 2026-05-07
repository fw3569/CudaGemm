#### 介绍
一个CUDA GEMM算子

#### 如何构建
构建依赖`CMake`、`nvcc`、`MSVC`和`Ninja`  
请自行更改或在命令行指定`CMAKE_CUDA_HOST_COMPILER`和`CMAKE_CXX_COMPILER`  
在`x64 Native Tools Command Prompt for VS 2022`中运行
```
cmake -G "Ninja Multi-Config" -S ./ -B build
cmake --build ./build --config RelWithDebInfo
```

#### 技术特性
分块优化：结合Shared Memory与Register Blocking分块，减少Global Memory访存，提升FFMA计算密度。  
地址对齐：使用alignas(128)确保首地址对齐缓存行，支持向量化指令提高访存效率。  
合并访存：通过调整访问模式合并访存，有效利用带宽。  
向量化访存：用float4向量化访存减少mio指令数量和地址计算开销。  
分支避免：用乘法运算代替分支，对编译器指令重排更友好。  
数据重排：通过重排数据存储位置，精确控制warp内bank分配，消除bank conflict。  
硬件资源约束：使用launch_bounds控制寄存器用量，确保维持活跃Warp数。

#### 性能测试

| 指标 | 参数 |
| :--- | :--- |
| **GPU** | NVIDIA GeForce MX450 |
| **Compute Capability** | 7.5 (Turing) |
| **SM 数量** | 14 |
| **显存带宽** | 80 GB/s |
| **理论算力** | 2.5 TFLOPS |

在`mx450`上测试$2048^3$的GEMM计算性能达到`cublasSgemm`的约`93%`，用`RelWithDebInfo`编译

```
kernel time : 8.57544ms, total time: 22.3206ms, tflops: 2.00338
cublas kernel time: 8.02606ms, rate: 0.935936
```

下图是python下调用dll测试benchmark的结果,用`-O3`和`/O2`编译

![benchmark results](./benchmark/benchmark_results.png)

下面是nsight的测试结果

![GPU Speed Of Light Throughput](./document/img/屏幕截图%202026-05-08%20115712.png)

![Roofline](./document/img/屏幕截图%202026-05-08%20115739.png)

位于Roofline模型右侧处于计算瓶颈，达到83.43%的计算吞吐量

![Occupancy](./document/img/屏幕截图%202026-05-08%20115816.png)

![Impact of Varying Register Count Per Thread](./document/img/屏幕截图%202026-05-08%20115856.png)

![Impact of Varying Shared Memory Usage Per Block](./document/img/屏幕截图%202026-05-08%20115917.png)

__launch_bounds__限制后，寄存器和共享内存共同限制了Occupancy，active warp数量达到理论值的97%，有效掩盖访存延迟

![Warp State](./document/img/屏幕截图%202026-05-08%20120054.png)

![Instruction Stall](./document/img/屏幕截图%202026-05-08%20120156.png)

Math Pipe Throttle达到33.63%，证明运算单元处于满载排队状态
