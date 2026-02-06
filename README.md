#### 介绍
一个CUDA GEMM算子

#### 如何构建
构建依赖`nvcc`和`msvc`  
请自行更改`CMakeLists`中的`CMAKE_CUDA_HOST_COMPILER`后编译`cmake --build ./build`

#### 性能测试
在`mx450`上测试性能达到约`cublasSgemm`的`200%`

```
kernel time: 1.9119ms, total time: 5.1912ms, tflops: 1.12322
cublas kernel time: 4.40349ms, rate: 2.3032
```
