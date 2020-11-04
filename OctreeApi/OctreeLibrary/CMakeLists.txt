##########################################################################
# OctreeApi                                                          #
# A Library for generating octree structures for point clouds on the GPU #
##########################################################################

project(OctreeLibrary LANGUAGES CUDA CXX)

add_library( ${PROJECT_NAME} STATIC
        src/tools.cu
        src/denseOctree/cellMerging.cu
        src/denseOctree/initialPointCounting.cu
        src/denseOctree/pointDistribution.cu
        src/sparseOctree/cellMergingSparse.cu
        src/sparseOctree/initialPointCountingSparse.cu
        src/sparseOctree/pointDistributionSparse.cu
        include/octreeBase.h
        include/sparseOctree.h
        src/sparseOctree/sparseOctree.cpp include/denseOctree.h src/denseOctree/denseOctree.cpp src/sparseOctree/indexing.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src)

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