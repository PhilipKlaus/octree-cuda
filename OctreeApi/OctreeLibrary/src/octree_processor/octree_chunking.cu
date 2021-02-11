/**
 * @file octree_chunking.cu
 * @author Philip Klaus
 * @brief Contains implementations of chunking related methods
 */

#include "octree_processor.h"
#include "kernel_executor.cuh"
#include "point_counting.cuh"


void OctreeProcessor::initialPointCounting ()
{
    // Allocate the dense point count
    itsDensePointCountPerVoxel = createGpuU32 (itsMetadata.nodeAmountDense, "DensePointCountPerVoxel");
    itsDensePointCountPerVoxel->memset (0);

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = createGpuI32 (itsMetadata.nodeAmountDense, "DenseToSparseLUT");
    itsDenseToSparseLUT->memset (-1);

    // Allocate the temporary sparseIndexCounter
    auto nodeAmountSparse = createGpuU32 (1, "nodeAmountSparse");
    nodeAmountSparse->memset (0);

    auto& meta                       = itsCloud->getMetadata ();
    Kernel::KernelConfig config      = {meta.cloudType, meta.pointAmount};
    KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), meta.pointAmount, meta.pointDataStride};
    KernelStructs::Gridding gridding = {itsOctreeData->getGridSize (0), meta.cubicSize (), meta.bbCubic.min};

    float time = Kernel::pointCounting (
            config,
            itsDensePointCountPerVoxel->devicePointer (),
            nodeAmountSparse->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            cloud,
            gridding);

    // Store the current amount of sparse nodes
    // !IMPORTANT! At this time nodeAmountSparse holds just the amount of nodes
    // in the lowest level
    itsMetadata.nodeAmountSparse = nodeAmountSparse->toHost ()[0];
    itsTimeMeasurement.emplace_back ("initialPointCount", time);
    spdlog::info ("'pointCounting' took {:f} [ms]", time);
}