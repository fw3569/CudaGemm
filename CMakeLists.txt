cmake_minimum_required(VERSION 4.0)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CUDA_HOST_COMPILER "D:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64\\cl.exe")
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER_FORCED ON)
project(Proj LANGUAGES CXX CUDA)
add_executable(
  proj
  main.cu
)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(proj PRIVATE CUDA::cublas)
