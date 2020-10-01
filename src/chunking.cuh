//#pragma once
#ifndef OCTREE_CHUNKING
#define OCTREE_CHUNKING

#include "tools.cuh"
#include "types.h"

unique_ptr<CudaArray<uint32_t>> initialPointCounting(unique_ptr<CudaArray<Point>> pointCloud, uint32_t gridSize, Vector3 posOffset, Vector3 size, Vector3 minimum);

#endif