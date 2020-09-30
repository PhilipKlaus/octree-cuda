project(Tests)

set(TEST_SOURCES
        ../src/tools.cu
        )

add_executable(test-OctreeCuda main.cpp test_generat_equally_sampled.cpp ${TEST_SOURCES})
target_include_directories(test-OctreeCuda PRIVATE ${PROJECT_SOURCE_DIR}/External/)

set_target_properties(test-OctreeCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_test(NAME unittests-OctreeCuda COMMAND $<TARGET_FILE:test-OctreeCuda>)


