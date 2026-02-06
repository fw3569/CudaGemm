#### 介绍
一个CUDA GEMM算子

#### 如何构建
构建依赖`nvcc`和`msvc`  
请自行更改`CMakeLists`中的`CMAKE_CUDA_HOST_COMPILER`后编译`cmake --build ./build`

#### 性能测试
在`mx450`上测试性能达到约`cublasSgemm`的`60%`

```
kernel time : 1.51182ms, total time: 4.6995ms, tflops: 1.39938
cublas kernel time: 0.939987ms, rate: 0.621757
```
