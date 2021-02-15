/**
 * @file octree_chunking.cu
 * @author Philip Klaus
 * @brief Contains implementations of chunking-related Octreeprocessor methods
 */

#include "kernel_executor.cuh"
#include "octree_processpr_impl.cuh"

#include "hierarchical_merging.cuh"
#include "octree_initialization.cuh"
#include "point_count_propagation.cuh"
#include "point_counting.cuh"
#include "point_distributing.cuh"


void OctreeProcessorPimpl::OctreeProcessorImpl::initialPointCounting ()
{
    // Allocate the dense point count
    itsDensePointCountPerVoxel = createGpuU32 (itsMetadata.nodeAmountDense, "DensePointCountPerVoxel");
    itsDensePointCountPerVoxel->memset (0);

    // Allocate the conversion LUT from dense to sparse
    itsDenseToSparseLUT = createGpuI32 (itsMetadata.nodeAmountDense, "DenseToSparseLUT");
    itsDenseToSparseLUT->memset (-1);

    // Allocate the temporary sparseIndexCounter
    itsTmpCounting = createGpuU32 (1, "nodeAmountSparse");
    itsTmpCounting->memset (0);

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

    itsTimeMeasurement.emplace_back ("kernelPointCounting", time);
    spdlog::info ("[kernel] kernelPointCounting took {:f} [ms]", time);
}

void OctreeProcessorPimpl::OctreeProcessorImpl::performCellMerging ()
{
    float timeAccumulated = 0;

    // Perform a hierarchicaly merging of the grid cells which results in an octree structure
    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        float time = executeKernel (
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

        itsTimeMeasurement.emplace_back (
                "kernelPropagatePointCounts_" + std::to_string (itsOctreeData->getGridSize (i)), time);
        timeAccumulated += time;
    }

    spdlog::info ("[kernel] kernelPropagatePointCounts took {:f}[ms]", timeAccumulated);

    // Retrieve the actual amount of sparse nodes in the octree and allocate the octree data structure
    itsMetadata.nodeAmountSparse = itsTmpCounting->toHost ()[0];
    itsOctreeData->createOctree (itsMetadata.nodeAmountSparse);
    // Allocate the conversion LUT from sparse to dense
    itsSparseToDenseLUT = createGpuI32 (itsMetadata.nodeAmountSparse, "sparseToDenseLUT");
    itsSparseToDenseLUT->memset (-1);

    initLowestOctreeHierarchy ();
    mergeHierarchical ();
}

void OctreeProcessorPimpl::OctreeProcessorImpl::initLowestOctreeHierarchy ()
{
    float time = executeKernel (
            chunking::kernelInitLeafNodes,
            itsOctreeData->getNodes (0),
            itsOctreeData->getDevice (),
            itsDensePointCountPerVoxel->devicePointer (),
            itsDenseToSparseLUT->devicePointer (),
            itsSparseToDenseLUT->devicePointer (),
            itsOctreeData->getNodes (0));

    itsTimeMeasurement.emplace_back ("kernelInitLeafNodes", time);
    spdlog::info ("[kernel] kernelInitLeafNodes took {:f}[ms]", time);
}


void OctreeProcessorPimpl::OctreeProcessorImpl::mergeHierarchical ()
{
    itsTmpCounting->memset (0);

    float timeAccumulated = 0;
    for (uint32_t i = 0; i < itsMetadata.depth; ++i)
    {
        float time = executeKernel (
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

        timeAccumulated += time;
        itsTimeMeasurement.emplace_back (
                "kernelMergeHierarchical_" + std::to_string (itsOctreeData->getGridSize (i)), time);
    }

    spdlog::info ("[kernel] kernelMergeHierarchical took {:f}[ms]", timeAccumulated);
}

void OctreeProcessorPimpl::OctreeProcessorImpl::distributePoints ()
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

    itsTimeMeasurement.emplace_back ("kernelDistributePoints", time);
    spdlog::info ("[kernel] kernelDistributePoints took {:f}[ms]", time);
}
