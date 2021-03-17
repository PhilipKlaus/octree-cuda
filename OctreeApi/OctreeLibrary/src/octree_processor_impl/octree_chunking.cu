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
    Kernel::KernelConfig config      = {meta.cloudType, meta.pointAmount, "kernelPointCounting"};
    KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), meta.pointAmount, meta.pointDataStride};
    KernelStructs::Gridding gridding = {itsOctreeData->getGridSize (0), meta.cubicSize (), meta.bbCubic.min};

    Kernel::pointCounting (
            config,
            itsCountingGrid->devicePointer (),
            itsTmpCounting->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            cloud,
            gridding);
}

void OctreeProcessor::OctreeProcessorImpl::performCellMerging ()
{
    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        executeKernel (
                chunking::kernelPropagatePointCounts,
                itsOctreeData->getNodes (i + 1),
                "kernelPropagatePointCounts",
                itsCountingGrid->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsTmpCounting->devicePointer (),
                itsOctreeData->getNodes (i + 1),
                itsOctreeData->getGridSize (i + 1),
                itsOctreeData->getGridSize (i),
                itsOctreeData->getNodeOffset (i + 1),
                itsOctreeData->getNodeOffset (i));
    }

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
    executeKernel (
            chunking::kernelInitLeafNodes,
            itsOctreeData->getNodes (0),
            "kernelInitLeafNodes",
            itsOctreeData->getDevice (),
            itsCountingGrid->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            itsSparseToDenseLUT->devicePointer (),
            itsOctreeData->getNodes (0));
}


void OctreeProcessor::OctreeProcessorImpl::mergeHierarchical ()
{
    itsTmpCounting->memset (0);

    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        executeKernel (
                chunking::kernelMergeHierarchical,
                itsOctreeData->getNodes (i + 1),
                "kernelMergeHierarchical",
                itsOctreeData->getDevice (),
                itsCountingGrid->devicePointer (),
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
}

void OctreeProcessor::OctreeProcessorImpl::distributePoints ()
{
    auto tmpIndexRegister = createGpuU32 (itsMetadata.nodeAmountSparse, "tmpIndexRegister");
    tmpIndexRegister->memset (0);

    auto& meta                       = itsCloud->getMetadata ();
    Kernel::KernelConfig config      = {meta.cloudType, meta.pointAmount, "kernelDistributePoints"};
    KernelStructs::Cloud cloud       = {itsCloud->getCloudDevice (), meta.pointAmount, meta.pointDataStride};
    KernelStructs::Gridding gridding = {itsOctreeData->getGridSize (0), meta.cubicSize (), meta.bbCubic.min};

    Kernel::distributePoints (
            config,
            itsOctreeData->getDevice (),
            itsPointLut->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            tmpIndexRegister->devicePointer (),
            cloud,
            gridding);
}
