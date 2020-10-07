cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(OctreeCuda LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_STANDARD 14)


# find all the cuda headers
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/External")
include_directories(${CMAKE_CURRENT_BINARY_DIR})


set(SOURCES
        src/main.cpp
        src/tools.cu
        src/chunking.cu
        src/pointcloud.cpp
        )
        
add_executable(OctreeCuda ${SOURCES})

target_include_directories(OctreeCuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)


# link to cuda 
target_link_libraries(OctreeCuda
        PUBLIC ${CUDA_nppicc_LIBRARY}
        PUBLIC ${CUDA_nppig_LIBRARY}
        )
        
set_target_properties(OctreeCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        
target_compile_definitions(OctreeCuda PUBLIC CUDA_SUPPORT)

include(CTest)
option(BUILD_TESTS "build automated tests" ON)
if (BUILD_TESTS)
    message(STATUS "icihw build includes automated tests...")
    add_subdirectory(Tests)
endif()
