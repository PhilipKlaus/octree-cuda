##########################################################################
# OctreeApi                                                              #
# A Library for generating octree structures for point clouds on the GPU #
##########################################################################

project(OctreeLibrary LANGUAGES CUDA CXX)

add_library( ${PROJECT_NAME} STATIC
        src/tools.cu
        src/sparseOctree/sparseOctree.cpp
        src/sparseOctree/indexing.cpp

        src/timing.cuh

        # Sparse octree
        src/sparseOctree/benchmarking.cpp
        src/sparseOctree/data_export.cpp
        src/sparseOctree/tools.cpp
        src/sparseOctree/debugging.cpp

        # Cuda kernels
        src/kernel/initial_point_counting.cu
        src/kernel/point_count_propagation.cu
        src/kernel/octree_initialization.cu
        src/kernel/point_distributing.cu
        src/kernel/tree_node_merging.cu
        src/kernel/pseudo_random_subsampling.cu)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src src/kernel)

####################
# ADD CUDA SUPPORT #
####################

# find all the cuda headers
find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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


######################
# ADD SPDLOG LIBRARY #
######################
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)


#######################
# BUILD LIBRARY TESTS #
#######################
include(CTest)
option(BUILD_TESTS "build automated tests" ON)
if (BUILD_TESTS)
    message(STATUS "OctreeCuda build includes automated tests...")
    add_subdirectory(Tests)
endif()
