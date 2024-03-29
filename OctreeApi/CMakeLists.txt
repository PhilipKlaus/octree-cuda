##########################################################################
# OctreeApi                                                          #
# A Library for generating octree structures for point clouds on the GPU #
##########################################################################

project(OctreeApi LANGUAGES CUDA CXX)

add_library(${PROJECT_NAME} SHARED
        src/octreeApi.cpp
        src/session.h
        src/session.cpp src/json_exporter.h
        src/json_exporter.h)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src)

####################
# ADD CUDA SUPPORT #
####################

# find all the cuda headers
find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../External)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})


target_link_libraries(${PROJECT_NAME} PRIVATE project_warnings)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_SUPPORT=0)

############################
# Add Lohmann JSON library #
############################
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/External/nlohmann)

######################
# ADD SPDLOG LIBRARY #
######################
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog_header_only)

add_subdirectory(OctreeLibrary)
target_link_libraries(${PROJECT_NAME} PUBLIC OctreeLibrary)


