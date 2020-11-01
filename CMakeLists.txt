cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PotreeConverterGPU LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_STANDARD 14)

# Specify the PotreeConverterGPU executable
set(SOURCES src/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/External )

# Add the spdlog library
add_subdirectory(External/spdlog)
if(NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()


# Add the OctreeLib library
add_subdirectory(OctreeLibrary)


# Link all added libraries
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${PROJECT_NAME} PRIVATE OctreeLibrary)

if(WIN32)
    add_custom_command(TARGET PotreeConverterGPU
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:OctreeLibrary> "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Copy dll file to ${CMAKE_CURRENT_BINARY_DIR} directory" VERBATIM
            )
endif()
