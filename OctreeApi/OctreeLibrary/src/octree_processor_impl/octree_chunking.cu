/**
 * @file octree_chunking.cu
 * @author Philip Klaus
 * @brief Contains implementations of chunking-related OctreeProcessorImpl methods
 */

#include "kernel_executor.cuh"
#include "octree_processor_impl.cuh"
#include "time_tracker.cuh"

#include "hierarchical_merging.cuh"
#include "octree_initialization.cuh"
#include "point_count_propagation.cuh"
#include "point_counting.cuh"
#include "point_distributing.cuh"


void OctreeProcessor::OctreeProcessorImpl::initialPointCounting ()
{
    auto& meta                       = itsCloud->getMetadata ();
    Kernel::KernelConfig config      = {meta.cloudType, meta.pointAmount};
    KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), meta.pointAmount, meta.pointDataStride};
    KernelStructs::Gridding gridding = {itsOctreeData->getGridSize (0), meta.cubicSize (), meta.bbCubic.min};

    float time = Kernel::pointCounting (
            config,
            itsDensePointCountPerVoxel->devicePointer (),
            itsTmpCounting->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            cloud,
            gridding);

    TimeTracker::getInstance ().trackKernelTime (time, "kernelPointCounting");
}

void OctreeProcessor::OctreeProcessorImpl::performCellMerging ()
{
    float timeAccumulated = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        timeAccumulated += executeKernel (
                chunking::kernelPropagatePointCounts,
                itsOctreeData->getNodes (i + 1),
                itsDensePointCountPerVoxel->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsTmpCounting->devicePointer (),
                itsOctreeData->getNodes (i + 1),
                itsOctreeData->getGridSize (i + 1),
                itsOctreeData->getGridSize (i),
                itsOctreeData->getNodeOffset (i + 1),
                itsOctreeData->getNodeOffset (i));
    }

    TimeTracker::getInstance ().trackKernelTime (timeAccumulated, "kernelPropagatePointCounts");

    // Retrieve the actual amount of sparse nodes in the octree and allocate the octree data structure
    itsMetadata.nodeAmountSparse = itsTmpCounting->toHost ()[0];
    itsOctreeData->createOctree (itsMetadata.nodeAmountSparse);
    // Allocate the conversion LUT from sparse to dense
    itsSparseToDenseLUT = createGpuI32 (itsMetadata.nodeAmountSparse, "sparseToDenseLUT");
    itsSparseToDenseLUT->memset (-1);

    initLowestOctreeHierarchy ();
    mergeHierarchical ();
}

void OctreeProcessor::OctreeProcessorImpl::initLowestOctreeHierarchy ()
{
    float time = executeKernel (
            chunking::kernelInitLeafNodes,
            itsOctreeData->getNodes (0),
            itsOctreeData->getDevice (),
            itsDensePointCountPerVoxel->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            itsSparseToDenseLUT->devicePointer (),
            itsOctreeData->getNodes (0));

    TimeTracker::getInstance ().trackKernelTime (time, "kernelInitLeafNodes");
}


void OctreeProcessor::OctreeProcessorImpl::mergeHierarchical ()
{
    itsTmpCounting->memset (0);

    float timeAccumulated = 0.f;
    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        timeAccumulated += executeKernel (
                chunking::kernelMergeHierarchical,
                itsOctreeData->getNodes (i + 1),
                itsOctreeData->getDevice (),
                itsDensePointCountPerVoxel->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsSparseToDenseLUT->devicePointer (),
                itsTmpCounting->devicePointer (),
                itsMetadata.mergingThreshold,
                itsOctreeData->getNodes (i + 1),
                itsOctreeData->getGridSize (i + 1),
                itsOctreeData->getGridSize (i),
                itsOctreeData->getNodeOffset (i + 1),
                itsOctreeData->getNodeOffset (i));
    }
    TimeTracker::getInstance ().trackKernelTime (timeAccumulated, "kernelMergeHierarchical");
}

void OctreeProcessor::OctreeProcessorImpl::distributePoints ()
{
    auto tmpIndexRegister = createGpuU32 (itsMetadata.nodeAmountSparse, "tmpIndexRegister");
    tmpIndexRegister->memset (0);

    auto& meta                       = itsCloud->getMetadata ();
    Kernel::KernelConfig config      = {meta.cloudType, meta.pointAmount};
    KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), meta.pointAmount, meta.pointDataStride};
    KernelStructs::Gridding gridding = {itsOctreeData->getGridSize (0), meta.cubicSize (), meta.bbCubic.min};

    float time = Kernel::distributePoints (
            config,
            itsOctreeData->getDevice (),
            itsLeafLut->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            tmpIndexRegister->devicePointer (),
            cloud,
            gridding);

    TimeTracker::getInstance ().trackKernelTime (time, "kernelDistributePoints");
}
