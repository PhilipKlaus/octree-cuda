##########################################################################
# OctreeApi                                                              #
# A Library for generating octree structures for point clouds on the GPU #
##########################################################################

project(OctreeLibrary LANGUAGES CUDA CXX)

add_library( ${PROJECT_NAME} STATIC

        # Misc
        src/misc/tools.cu

        # Sparse octree
        src/sparseOctree/sparseOctree.cu
        src/sparseOctree/benchmarking.cpp
        src/sparseOctree/data_export.cpp
        src/sparseOctree/tools.cpp
        src/sparseOctree/debugging.cpp

        # Cuda kernels

        src/kernel/point_counting.cuh
        src/kernel/octree_initialization.cuh
        src/kernel/point_count_propagation.cuh
        src/kernel/hierarchical_merging.cuh
        src/kernel/point_distributing.cuh
        src/kernel/random_subsampling.cuh
        src/kernel/first_point_subsampling.cuh
        src/kernel/subsample_evaluating.cuh)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src/kernel src/misc)

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
target_include_directories(${PROJECT_NAME} PRIVATE External/nlohmann)

######################
# Add LASZIP library #
######################
target_include_directories(${PROJECT_NAME} PRIVATE External/laszip)

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
