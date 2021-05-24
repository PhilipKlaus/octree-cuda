[![PhilipKlaus](https://circleci.com/gh/PhilipKlaus/octree-cuda.svg?style=svg&circle-token=80c3b9a5002f85c01d952c8f06abc9cfaaed5106)](https://app.circleci.com/pipelines/github/PhilipKlaus/octree-cuda)

# About

PotreeConverterGPU generates [Potree](https://github.com/potree/potree/) compatible LOD data structures from point clouds entirely on 
the GPU using CUDA. 

The project is part of a master's thesis with the title *In-core level-of-detail generation for point clouds
on GPUs using CUDA* and is conducted by the [Technical Universe of Vienna](https://www.cg.tuwien.ac.at/research/projects/Scanopy/) 
in corporation with the [AIT-Austrian Institute of Technology](https://www.ait.ac.at/en/).

# Project Status
This project is a research project!

| Feature                   | Status    | Comment                                                   |
|---------------------------|-----------|-----------------------------------------------------------|
| LOD Generation on GPU     | &#9989;   | Done                                                      |
| Exporting Potree data     | &#9989;   | Done                                                      |
| Unit Tests                | &#9745;   | In Progress...                                            |
| Source code documentation | &#9745;   | In Progress...                                            |
| Ply Import                | &#10060;  | Only prepared binary files can be imported and processed  |
| LAZ Import                | &#10060;  | Only prepared binary files can be imported and processed  |


# Release version
Be aware that the master branch is constantly updated. 
Therefore you should checkout or download release versions.
This releases also contain necessary input files (morrowbay.bin, heidentor.bin, coin.bin) .

# Getting started

## Building from source

### Prerequirements

| Name                            | Minimum Version   | Link                                                                                        |
| --------------------------------|-------------------| --------------------------------------------------------------------------------------------|
| CMAKE                           | 3.10              | [https://cmake.org/](https://cmake.org/)                                                    |
| CUDA                            | 11.2              | [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)  |
| Prepared point cloud files      | -                 | [Downloads](http://www.dreamcoder.at/potree/download.html)                                  |
| c++ 17 compiler                 | -                 | -                                                                                           |

### Building instructions

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
cmake --build . --config Release
```

**Note**:

* ``-DENABLE_KERNEL_TIMINGS=ON`` enables CUDA kernel timings (performance drawback)
* ``-DENABLE_KERNEL_ERROR_CHK=ON`` performs full error check after each CUDA kernel (performance drawback)

## Running PotreeConverterGPU

Put the binary file(s) in the same folder as the PotreeConverterGPU executable and run the program with the following
commands:

* morrowbay.bin

``PotreeConverterGPU.exe -f morrobay.bin -o .\output -p 119701547 -d 27,0.01 -t double -g 512,128 -a -r``

* heidentor.bin

``PotreeConverterGPU.exe -f heidentor.bin -o .\output -p 25836417 -d 15,0.001 -t float -g 512,128 -a -r``
* coin.bin

``PotreeConverterGPU.exe -f coin.bin -o .\output -p 5138448 -d 15,0.001 -t float -g 512,128 -a -r``

### Program Arguments
```
Usage:
  PotreeConverterGPU [OPTION...]

  -f, --file arg              File name point cloud
  -a, --averaging             Apply color averaging
  -r, --random                Perform Random-Subsampling, otherwise First-Point-Subsampling is applied
  -o, --output arg            Output path for the Potree data
  -p, --points arg            Point amount of the cloud
  -t, --type arg              The datatype of the cloud coordinates: "float" / "double"
  -d, --data arg              Data infos for stride and scale: [float, float]
  -g, --grids arg             Grid sizes for chunking and subsampling: [int, int]
  -m, --merge_threshold arg   The merging threshold (default: 10000)
  -e, --estimated_output arg  The estimated output point amount factor (default: 2.2)
  -h, --help                  Print usage
```

### Output
PotreeConverterGPU generates the following output files:

| Filename                  | Description                                                                           | 
| --------------------------|---------------------------------------------------------------------------------------|
| hierarchy.bin             | Contains information about octree nodes in binary form (required by Potree)           | 
| memory_report.html        | A diagram which shows the total GPU memory consumption per cudamalloc and cudafree    |
| metadata.json             | Octree metadata and data description (required by Potree)                             |
| octree.bin                | The binary lod cloud data (required by Potree)                                        |
| point_distribution.html   | A diagram which shows the point distribution in the leaf nodes                        |
| statistics.json           | Octree related statistics and information        

The resulting data can be directly rendered using [PotreeDesktop](https://github.com/potree/PotreeDesktop). 

# Project structure
-  [External/](External/) contains external tools and libraries
-  [OctreeApi/](OctreeApi/) Exposes the functionality from the OctreeLibrary using an C-API. The API is compiled to a shared library.
-  [OctreeApi/OctreeLibrary/](OctreeApi/OctreeLibrary) contains the actual LOD generation logic and CUDA kernels. It is compiled to a static library.
-  [src/](src/) contains the PotreeConverterGPU project. This project builds an executable and links to the OctreeApi.

# External Tools/Libraries
| Library           | Description               | Link                                      |
| ------------------|---------------------------|-------------------------------------------------------------------------------|
| Catch2            | Unit Testing framework    | [https://github.com/catchorg/Catch2](https://github.com/catchorg/Catch2)      |
| Cxxopts           | Command line parsing      | [https://github.com/jarro2783/cxxopts](https://github.com/jarro2783/cxxopts)  |
| Nlohmann          | JSON library              | [https://github.com/nlohmann/json](https://github.com/nlohmann/json)          |
| Spdlog            | Logging library           | [https://github.com/gabime/spdlog](https://github.com/gabime/spdlog)          |
