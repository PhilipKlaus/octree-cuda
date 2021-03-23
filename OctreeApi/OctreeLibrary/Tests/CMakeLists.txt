project(OctreeTest)

# Create the Test executable
add_executable(OctreeTest
        main.cu
        test_point_cloud_host.cu
        test_point_cloud_device.cu
        test_octree.cu
        # test_tools.cpp
        # test_initial_point_counting_sparse.cpp
        # test_merging_sparse.cpp
        # test_point_distribution_sparse.cpp
        # test_subsampling_sparse.cpp
        )

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

####################
# ADD CUDA SUPPORT #
####################

# find all the cuda headers
find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# link to cuda
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_SUPPORT)

target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${PROJECT_NAME} PRIVATE OctreeLibrary)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../../External)

get_target_property(private_include_dirs OctreeLibrary INCLUDE_DIRECTORIES)
get_target_property(private_link_libraries OctreeLibrary LINK_LIBRARIES)
target_include_directories(${PROJECT_NAME} PRIVATE ${private_include_dirs})
target_link_libraries(${PROJECT_NAME} PRIVATE ${private_link_libraries})


set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)

target_link_libraries(${PROJECT_NAME} PRIVATE project_warnings)
