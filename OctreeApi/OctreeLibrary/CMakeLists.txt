##########################################################################
# OctreeApi                                                              #
# A Library for generating octree structures for point clouds on the GPU #
##########################################################################

project(OctreeLibrary LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} STATIC

        # Public Headers
        include/octree_processor.h
        include/eventWatcher.h
        include/metadata.h
        include/defines.cuh

        # Private Headers
        src/include/cudaArray.h
        src/include/tools.cuh
        src/include/types.cuh
        src/include/kernel_structs.cuh
        src/include/octree.cuh
        src/point_cloud/point_cloud.cuh

        # CUDA Kernels
        src/kernel/kernel_executor.cuh
        src/kernel/hierarchical_merging.cuh
        src/kernel/octree_initialization.cuh
        src/kernel/point_count_propagation.cuh
        src/kernel/point_counting.cuh
        src/kernel/point_distributing.cuh
        src/kernel/random_subsampling.cuh
        src/kernel/subsample_evaluating.cuh
        src/kernel/kernel_helpers.cuh

        # OctreeProcessor
        src/octree_processor.cu
        src/include/octree_processor_impl.cuh
        src/octree_processor_impl/octree_subsampling.cu
        src/octree_processor_impl/benchmarking.cu
        src/octree_processor_impl/debugging.cu
        src/octree_processor_impl/octree_processor_impl.cu
        src/octree_processor_impl/dummy.cpp
        src/octree_processor_impl/octree_chunking.cu

        # Octree export
        src/include/octree_exporter.cuh
        src/include/ply_exporter.cuh
        src/include/potree_exporter.cuh
        src/octree_exporter/thread_pool.h
        src/octree_exporter/ply_exporter.cu
        src/octree_exporter/potree_exporter.cu

        # Octree data
        src/include/octree.cuh
        src/octree/octree.cu

        # Point cloud
        src/point_cloud/point_cloud.cuh
        src/point_cloud/point_cloud.cu

        # Misc
        src/misc/tools.cu
        src/include/timing.cuh)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE
        src/kernel
        src/misc
        src/include
        src/octree_exporter)

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

############################
# Add Lohmann JSON library #
############################
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/External/nlohmann)

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
endif ()
