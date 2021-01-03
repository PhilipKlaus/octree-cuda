project(OctreeTest)

# Create the Test executable
add_executable(OctreeTest
        main.cpp
        test_tools.cpp
        test_initial_point_counting_sparse.cpp
        test_merging_sparse.cpp
        test_point_distribution_sparse.cpp test_subsampling_sparse.cpp)

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

target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${PROJECT_NAME} PRIVATE OctreeLibrary)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../../External)



set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)


