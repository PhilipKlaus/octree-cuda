cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(OctreeCuda LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_STANDARD 14)

# Specify the OctreeCuda executable
set(SOURCES src/main.cpp)
add_executable(OctreeCuda ${SOURCES})

# find all the cuda headers
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/External")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add the spdlog library
add_subdirectory(External/spdlog)
if(NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()

# Add the OctreeLib library
add_subdirectory(lib)

# Link all added libraries
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${PROJECT_NAME} PRIVATE OctreeLib)

# link to cuda 
target_link_libraries(${PROJECT_NAME}
        PUBLIC ${CUDA_nppicc_LIBRARY}
        PUBLIC ${CUDA_nppig_LIBRARY}
        )
        
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        
target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_SUPPORT)

include(CTest)
option(BUILD_TESTS "build automated tests" ON)
if (BUILD_TESTS)
    message(STATUS "OctreeCuda build includes automated tests...")
    add_subdirectory(Tests)
endif()
