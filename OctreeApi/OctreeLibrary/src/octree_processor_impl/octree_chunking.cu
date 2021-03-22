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
    KernelStructs::Gridding gridding = {itsOctree->getGridSize (0), meta.cubicSize (), meta.bbCubic.min};

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
    for (uint32_t i = 0; i < itsOctree->getMetadata ().depth; ++i)
    {
        executeKernel (
                chunking::kernelPropagatePointCounts,
                itsOctree->getNodeAmount (i + 1),
                "kernelPropagatePointCounts",
                itsCountingGrid->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsTmpCounting->devicePointer (),
                itsOctree->getNodeAmount (i + 1),
                itsOctree->getGridSize (i + 1),
                itsOctree->getGridSize (i),
                itsOctree->getNodeOffset (i + 1),
                itsOctree->getNodeOffset (i));
    }

    // Retrieve the actual amount of sparse nodes in the octree and allocate the octree data structure
    uint32_t sparseNodes = itsTmpCounting->toHost ()[0];
    itsOctree->createHierarchy (sparseNodes);
    // Allocate the conversion LUT from sparse to dense
    itsSparseToDenseLUT = createGpuI32 (sparseNodes, "sparseToDenseLUT");
    itsSparseToDenseLUT->memset (-1);

    initLowestOctreeHierarchy ();
    mergeHierarchical ();
}

void OctreeProcessor::OctreeProcessorImpl::initLowestOctreeHierarchy ()
{
    executeKernel (
            chunking::kernelInitLeafNodes,
            itsOctree->getNodeAmount (0),
            "kernelInitLeafNodes",
            itsOctree->getDevice (),
            itsCountingGrid->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            itsSparseToDenseLUT->devicePointer (),
            itsOctree->getNodeAmount (0));
}


void OctreeProcessor::OctreeProcessorImpl::mergeHierarchical ()
{
    itsTmpCounting->memset (0);

    for (uint32_t i = 0; i < itsOctree->getMetadata ().depth; ++i)
    {
        executeKernel (
                chunking::kernelMergeHierarchical,
                itsOctree->getNodeAmount (i + 1),
                "kernelMergeHierarchical",
                itsOctree->getDevice (),
                itsCountingGrid->devicePointer (),
                itsDenseToSparseLUT->devicePointer (),
                itsSparseToDenseLUT->devicePointer (),
                itsTmpCounting->devicePointer (),
                itsOctree->getMetadata ().mergingThreshold,
                itsOctree->getNodeAmount (i + 1),
                itsOctree->getGridSize (i + 1),
                itsOctree->getGridSize (i),
                itsOctree->getNodeOffset (i + 1),
                itsOctree->getNodeOffset (i));
    }
}

void OctreeProcessor::OctreeProcessorImpl::distributePoints ()
{
    auto tmpIndexRegister = createGpuU32 (itsOctree->getMetadata ().nodeAmountSparse, "tmpIndexRegister");
    tmpIndexRegister->memset (0);

    auto& meta                  = itsCloud->getMetadata ();
    Kernel::KernelConfig config = {meta.cloudType, meta.pointAmount, "kernelDistributePoints"};
    KernelStructs::Cloud cloud  = {
            itsCloud->getCloudDevice (),
            meta.pointAmount,
            meta.pointDataStride,
            {
                    1.0 / meta.scale.x,
                    1.0 / meta.scale.y,
                    1.0 / meta.scale.z,
            }};
    KernelStructs::Gridding gridding = {itsOctree->getGridSize (0), meta.cubicSize (), meta.bbCubic.min};

    Kernel::distributePoints (
            config,
            itsOctree->getDevice (),
            itsPointLut->devicePointer (),
            itsCloud->getOutputBuffer_d (),
            itsDenseToSparseLUT->devicePointer (),
            tmpIndexRegister->devicePointer (),
            cloud,
            gridding);
}
