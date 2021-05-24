##########################################################################
# OctreeLibrary                                                          #
# A Library for generating octree structures for point clouds on the GPU #
##########################################################################

project(OctreeLibrary LANGUAGES CUDA)

add_library(${PROJECT_NAME} STATIC

        # Public Headers
        include/octree_processor.cuh
        include/memory_tracker.cuh
        include/time_tracker.cuh
        include/metadata.cuh
        include/defines.cuh
        include/timing.cuh

        # Private Headers
        src/include/tools.cuh
        src/include/types.cuh
        src/include/kernel_structs.cuh

        # CUDA Kernels
        src/kernel/kernel_executor.cuh
        src/kernel/hierarchical_merging.cuh
        src/kernel/octree_initialization.cuh
        src/kernel/point_count_propagation.cuh
        src/kernel/point_count_propagation.cuh
        src/kernel/point_counting.cuh
        src/kernel/point_distributing.cuh
        src/kernel/random_subsampling.cuh
        src/kernel/subsample_evaluating.cuh
        src/kernel/kernel_helpers.cuh
        src/kernel/random_initialization.cuh

        # OctreeProcessor
        src/octree_processor.cu
        src/include/octree_processor_impl.cuh
        src/octree_processor_impl/subsampling.cu
        src/octree_processor_impl/histogram.cu
        src/octree_processor_impl/octree_processor_impl.cu
        src/octree_processor_impl/octree_chunking.cu

        # Octree export
        src/include/octree_exporter.cuh
        src/include/ply_exporter.cuh
        src/include/potree_exporter.cuh
        src/octree_exporter/ply_exporter.cu
        src/octree_exporter/potree_exporter.cu

        # Octree data
        src/data/cuda_array.cuh
        src/include/octree.cuh
        src/data/octree.cu

        # Point cloud
        src/include/point_cloud.cuh
        src/data/point_cloud.cu

        # Misc
        src/misc/tools.cu)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE
        src/kernel
        src/misc
        src/include
        src/octree_exporter
        src/data)

###########
# OPTIONS #
###########
IF (ENABLE_KERNEL_TIMINGS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE KERNEL_TIMINGS)
    message(STATUS "Enable Cuda kernel timings")
ENDIF (ENABLE_KERNEL_TIMINGS)
IF (ENABLE_KERNEL_ERROR_CHK)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ERROR_CHECKS)
    message(STATUS "Enable Cuda kernel error checks")
ENDIF (ENABLE_KERNEL_ERROR_CHK)

####################
# ADD CUDA SUPPORT #
####################

# find all the cuda headers
find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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

###########################
# APPLY WARNINGS SETTINGS #
###########################
target_link_libraries(${PROJECT_NAME} PRIVATE project_warnings)

#######################
# BUILD LIBRARY TESTS #
#######################
include(CTest)
option(BUILD_UNIT_TESTS "build automated tests" ON)
if (BUILD_UNIT_TESTS)
    message(STATUS "OctreeCuda build includes automated tests...")
    add_subdirectory(Tests)
endif ()
