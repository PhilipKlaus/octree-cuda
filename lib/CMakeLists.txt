project(OctreeLib)

add_library( ${PROJECT_NAME} STATIC
        src/initialPointCounting.cu
        src/chunking.cu
        src/pointcloud.cpp
        src/tools.cu
        )

target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(${PROJECT_NAME}  PUBLIC CUDA_SUPPORT)
