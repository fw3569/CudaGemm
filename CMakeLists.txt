cmake_minimum_required(VERSION 4.0)
set(CMAKE_CUDA_HOST_COMPILER "D:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64\\cl.exe")
set(CMAKE_CXX_COMPILER "D:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64\\cl.exe")
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER_FORCED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
project(Proj LANGUAGES CXX CUDA)
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
add_executable(
  proj
  src/main.cpp
  src/gemm_kernel.cu
)
target_include_directories(proj PRIVATE ./include)
find_package(CUDAToolkit REQUIRED)
target_link_directories(proj PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(proj PRIVATE
    CUDA::cudart
    CUDA::cublas
    cudadevrt
)
target_compile_options(proj PRIVATE
$<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler /Zi>
)
target_link_options(proj PRIVATE
    $<$<CONFIG:RelWithDebInfo>:-Xlinker /DEBUG>
)
