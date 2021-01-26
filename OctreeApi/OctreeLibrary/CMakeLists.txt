##########################################################################
# OctreeApi                                                              #
# A Library for generating octree structures for point clouds on the GPU #
##########################################################################

project(OctreeLibrary LANGUAGES CUDA CXX)

add_library( ${PROJECT_NAME} STATIC

        # Headers
        include/api_types.h
        include/sparseOctree.h
        src/include/cudaArray.h
        src/include/eventWatcher.h
        include/octree_metadata.h
        src/include/tools.cuh
        include/types.cuh

        # Kernel
        src/kernel/kernel_executor.cuh
        src/kernel/first_point_subsampling.cuh
        src/kernel/hierarchical_merging.cuh
        src/kernel/octree_initialization.cuh
        src/kernel/point_count_propagation.cuh
        src/kernel/point_counting.cuh
        src/kernel/point_distributing.cuh
        src/kernel/random_subsampling.cuh
        src/kernel/subsample_evaluating.cuh

        # Misc
        src/misc/tools.cu
        include/defines.cuh
        src/misc/timing.cuh

        # Sparse octree
        src/sparseOctree/pipeline/first_point_subsampling.cu
        src/sparseOctree/pipeline/random_subsampling.cu
        src/sparseOctree/benchmarking.cu
        src/sparseOctree/debugging.cu
        src/sparseOctree/sparseOctree.cu
        src/sparseOctree/dummy.cpp

        # Data export
        src/export/octree_exporter.cuh
        src/export/ply_exporter.cuh
        src/export/ply_exporter.cu
        src/export/potree_exporter.cuh
        src/export/potree_exporter.cu)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE
        src/kernel
        src/misc
        src/include
        src/export)

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
endif()
