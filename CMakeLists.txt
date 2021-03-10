########################################################
# PotreeConverterGPU                                   #
# Converts a point cloud to a Potree compatible format #
########################################################
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PotreeConverterGPU LANGUAGES CUDA CXX)
find_package(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_STANDARD 14)

# Specify the PotreeConverterGPU executable
set(SOURCES
        src/main.cpp
        src/boundingbox.h)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/External ${PROJECT_SOURCE_DIR}/src)

#######################
# COMPILE DEFINITIONS #
#######################
OPTION(ENABLE_KERNEL_TIMINGS "Turn Cuda kernel timings on or off" OFF) # Disable by default
IF(ENABLE_KERNEL_TIMINGS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE KERNEL_TIMINGS)
ENDIF(ENABLE_KERNEL_TIMINGS)


######################
# ADD SPDLOG LIBRARY #
######################
add_subdirectory(External/spdlog)
if(NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()


######################
# ADD OCTREE LIBRARY #
######################
add_subdirectory(OctreeApi)


######################
# LINK TO LIBRARIES  #
######################
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${PROJECT_NAME} PRIVATE OctreeApi)


#########################################
# COPY OCTREE LIBRARY DLL TO BINARY DIR #
#########################################
if(WIN32)
    add_custom_command(TARGET PotreeConverterGPU
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:OctreeApi> "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Copy dll file to ${CMAKE_CURRENT_BINARY_DIR} directory" VERBATIM
            )
endif()
