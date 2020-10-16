project(Tests)

set(TEST_SOURCES
        ../src/tools.cu
        ../src/chunking.cu
        ../src/pointcloud.cpp

        test_tools.cpp)

add_executable(test-OctreeCuda
        main.cpp
        test_chunking.cpp
        test_tools.cpp
        ${TEST_SOURCES})
# target_include_directories(test-OctreeCuda PRIVATE ${PROJECT_SOURCE_DIR}/External/)
target_link_libraries(test-OctreeCuda PRIVATE spdlog::spdlog_header_only)

set_target_properties(test-OctreeCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME unittests-OctreeCuda COMMAND $<TARGET_FILE:test-OctreeCuda>)


