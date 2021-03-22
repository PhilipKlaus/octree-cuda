/**
 * @file octree_processor_impl.cuh
 * @author Philip Klaus
 * @brief Contains the implementation of the OctreeProcessorImpl
 */
#pragma once

#include "octree.cuh"
#include "octree_processor.cuh"
#include "point_cloud.cuh"
#include "types.cuh"


class OctreeProcessor::OctreeProcessorImpl
{
public:
    OctreeProcessorImpl (
            uint8_t* pointCloud,
            uint32_t chunkingGrid,
            uint32_t mergingThreshold,
            PointCloudMetadata cloudMetadata,
            SubsampleMetadata subsamplingMetadata);

    OctreeProcessorImpl (const OctreeProcessorImpl&) = delete;
    void operator= (const OctreeProcessorImpl&) = delete;

public:
    ///@{
    ///@name Chunking
    void initialPointCounting ();
    void performCellMerging ();
    void distributePoints ();
    void performSubsampling ();
    ///@}

    ///@{
    ///@name Exporting
    void exportHistogram (const string& filePath, uint32_t binWidth);
    void exportPlyNodes (const string& folderPath);
    void exportPotree (const string& folderPath);
    ///@}

    const NodeStatistics& getNodeStatistics ();
    const OctreeMetadata& getMetadata () const;
    void updateOctreeStatistics ();

private:
    void mergeHierarchical ();
    void initLowestOctreeHierarchy ();
    void prepareSubsampleConfig (SubsampleSet& subsampleSet, uint32_t parentIndex);
    void calculateVoxelBB (PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level);

    void randomSubsampling (const unique_ptr<int[]>& h_sparseToDenseLUT, uint32_t sparseVoxelIndex, uint32_t level);

    void histogramBinning (
            const shared_ptr<Chunk[]>& h_octreeSparse,
            std::vector<uint32_t>& counts,
            uint32_t min,
            uint32_t binWidth,
            uint32_t nodeIndex) const;

    void setActiveParent (uint32_t parentNode);
    int getLastParent ();


private:
    PointCloud itsCloud;
    std::unique_ptr<Octree> itsOctree;

    SubsampleMetadata itsSubsampleMetadata;

    // Required helper data structures for calculation
    GpuArrayU32 itsCountingGrid;
    GpuArrayI32 itsDenseToSparseLUT;
    GpuArrayI32 itsSparseToDenseLUT;
    GpuArrayU32 itsTmpCounting;
    GpuPointLut itsPointLut;
    GpuAveraging itsAveragingGrid;
    GpuRandomState itsRandomStates;
    GpuArrayU32 itsRandomIndices;
    int itsLastSubsampleNode;
};
