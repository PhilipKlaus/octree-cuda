########################################################
# PotreeConverterGPU                                   #
# Converts a point cloud to a Potree compatible format #
########################################################
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PotreeConverterGPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

######################
# WARNINGS INTERFACE #
######################
add_library(project_warnings INTERFACE)
if (MSVC)
    string(REGEX REPLACE "/W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    target_compile_options(project_warnings INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W4 /WX>)
    target_compile_options(project_warnings INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler> -WX)
else ()
    target_compile_options(project_warnings INTERFACE $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall -Wextra -Werror>)
    target_compile_options(project_warnings INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Werror>)
endif ()

# Specify the PotreeConverterGPU executable
set(SOURCES
        src/main.cpp
        src/boundingbox.h
        src/argparser.h)

add_executable(${PROJECT_NAME} ${SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/External ${PROJECT_SOURCE_DIR}/src)

###########
# OPTIONS #
###########
OPTION(ENABLE_KERNEL_TIMINGS "Turn Cuda kernel timings on or off" OFF) # Disable by default
OPTION(ENABLE_KERNEL_ERROR_CHK "Turn Cuda kernel error checks on or off" OFF) # Disable by default


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
target_link_libraries(${PROJECT_NAME} PRIVATE project_warnings)


#########################################
# COPY OCTREE LIBRARY DLL TO BINARY DIR #
#########################################
if(WIN32)
    add_custom_command(TARGET PotreeConverterGPU
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:OctreeApi> $<TARGET_FILE_DIR:${PROJECT_NAME}>
            COMMENT "Copy dll file to output directory" VERBATIM
            )
endif()
