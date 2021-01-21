#pragma once

#include "types.cuh"

class PlyExporter
{
public:
    PlyExporter (const GpuArrayU8& pointCloud,
                 const GpuOctree& octree,
                 const GpuArrayU32 leafeLut,
                 const unordered_map<uint32_t, GpuArrayU32>& parentLut,
                 const unordered_map<uint32_t, GpuAveraging>& parentAveraging);



};
