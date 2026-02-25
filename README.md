#### 介绍
一个CUDA GEMM算子

#### 如何构建
构建依赖`nvcc`和`msvc`  
请自行更改`CMakeLists`中的`CMAKE_CUDA_HOST_COMPILER`后编译`cmake --build ./build`

#### 性能测试
在`mx450`上测试$1024^3$的GEMM计算性能达到`cublasSgemm`的约`85%`，用`RelWithDebInfo`编译

```
kernel time : 1.21027ms, total time: 4.50247ms, tflops: 1.4979
cublas kernel time: 1.03283ms, rate: 0.853382
```

下图是python下调用dll测试的结果,用`-O3`和`/O2`编译

![Figure 1](./benchmark/benchmark_results.png)
