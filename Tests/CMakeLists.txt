project(OctreeTest)

# Create the Test executable
add_executable(OctreeTest
        main.cpp
        test_merging.cpp
        test_tools.cpp
        test_initial_point_counting.cpp
        test_point_distributing.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${PROJECT_NAME} PRIVATE OctreeLib)


set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME ${PROJECT_NAME} COMMAND $<TARGET_FILE:${PROJECT_NAME}>)


