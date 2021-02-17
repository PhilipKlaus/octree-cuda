# octree-cuda

The repository contains a project for generating a [Potree](https://github.com/potree/potree/) compatible LOD data structure on the GPU.
The entire algorihmics are implemented using CUDA.

## Project structure
-  [External/](External/) contains external librarys such as [catch2](https://github.com/catchorg/Catch2) and [spdlog](https://github.com/gabime/spdlog)
-  [OctreeApi/OctreeLibrary/](OctreeApi/OctreeLibrary) contains the actual LOD generation logic and CUDA kernels. It is compiled to a static library.
-  [OctreeApi/](OctreeApi/) Exposes the functionality from the OctreeLibrary using an C-API. The API is compiled to a shared library.
-  [src/](src/) contains the PotreeConverterGPU project. This project builds an executable and links to the OctreeApi.

## OctreeApi

The OctreeApi calls the OctreeLibrary and uses a session-based state saving model. 
Using this kind of architecture multiple sessions can be instantiated and work in parallel.
Furthermore, the API gathers information and metadata from the OctreeLibrary and exports them using the [json_exporter](OctreeApi/src/json_exporter.h)

## OctreeLibrary

-  [OctreeApi/OctreeLibrary/include/](OctreeApi/OctreeLibrary/include/) Contains public include files. These files are used by the OctreeApi. 
The file [octree_processor.cuh](https://github.com/PhilipKlaus/octree-cuda/blob/master/OctreeApi/OctreeLibrary/include/octree_processor.cuh) exposes the main LOD functionality. It uses a PIMPL pattern to hide its implementation details which can be found in [octree_processor_impl.cuh](https://github.com/PhilipKlaus/octree-cuda/blob/master/OctreeApi/OctreeLibrary/src/include/octree_processor_impl.cuh) (declarations) and in the folder [OctreeApi/OctreeLibrary/src/octree_processor_impl/](OctreeApi/OctreeLibrary/src/octree_processor_impl) (implementation).

- The CUDA kernels can be found in [OctreeApi/OctreeLibrary/src/kernel](OctreeApi/OctreeLibrary/src/kernel)
