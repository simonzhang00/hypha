cmake_minimum_required(VERSION 2.8)

project("hypha")


find_package(CUDA 7 REQUIRED)

#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

INCLUDE(CheckCXXSourceCompiles)

include_directories (include)
include_directories(${PROJECT_SOURCE_DIR}/deps/cub)


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-g;-G;-Xcompiler;-ggdb;-std=c++11)
else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-O3;-Xcompiler;-fopenmp;-std=c++11;-w;-DNDEBUG)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
#endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

FIND_PACKAGE(OpenMP)

set (CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
#include <omp.h>
int main() {
#if (_OPENMP >= 200805 || _MSC_VER >= 1500)
  return 0;
#else
  breaks_on_purpose
#endif
}
" OPENMP_VERSION)

if(OPENMP_VERSION)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
message("

OpenMP 3.0 not supported by the compiler (${CMAKE_CXX_COMPILER})! 
To get optimal performance for the \"chunk\" algorithm, use a C++ compiler with OpenMP 3.0 support (e.g., GCC >=4.4).
To use a different compiler, pass it to cmake in the variable CMAKE_CXX_COMPILER: 
 cmake . -DCMAKE_CXX_COMPILER=g++-4.7

")
endif()


FILE(GLOB_RECURSE all_headers "include/*.h")

FILE(GLOB general_includes_phat "include/phat/*.h")
FILE(GLOB algorithms_includes_phat "include/phat/algorithms/*.h")
FILE(GLOB helpers_includes_phat "include/phat/helpers/*.h")
FILE(GLOB representations_includes_phat "include/phat/representations/*.h")

FILE(GLOB general_includes_hypha "include/hypha/*.h")
FILE(GLOB algorithms_includes_hypha "include/hypha/algorithms/*.h")
FILE(GLOB helpers_includes_hypha "include/hypha/helpers/*.h")

add_executable(info src/info.cpp ${general_includes_phat})
add_executable(phat src/phat.cpp ${general_includes_phat})
add_executable(convert src/convert.cpp ${all_headers})
cuda_add_executable(hypha src/main.cu ${all_headers})

target_link_libraries(hypha ${EXTRA_LIBS})

source_group(Header\ Files_phat FILES ${general_includes_phat})
source_group(Header\ Files_phat\\helpers FILES ${helpers_includes_phat})
source_group(Header\ Files_phat\\representations FILES ${representations_includes_phat})
source_group(Header\ Files_phat\\algorithms FILES ${algorithms_includes_phat})

source_group(Header\ Files_hypha FILES ${general_includes_hypha})
source_group(Header\ Files_hypha\\algorithms FILES ${algorithms_includes_hypha})
source_group(Header\ Files_hypha\\helpers FILES ${helpers_includes_hypha})
