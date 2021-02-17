/**
 * @file octree_processor_impl.cuh
 * @author Philip Klaus
 * @brief Contains the implementation of the OctreeProcessorImpl
 */
#pragma once

#include "octree.cuh"
#include "octree_processor.cuh"
#include "point_cloud.cuh"
#include "subsampling_data.cuh"
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
    ///@name Chunking
    void exportHistogram (const string& filePath, uint32_t binWidth);
    void exportPlyNodes (const string& folderPath);
    void exportPotree (const string& folderPath);
    ///@}

    const OctreeMetadata& getMetadata () const;

    void updateOctreeStatistics ();
    unique_ptr<uint32_t[]> getDataLUT () const;
    unique_ptr<uint32_t[]> getDensePointCountPerVoxel () const;
    unique_ptr<int[]> getDenseToSparseLUT () const;
    unique_ptr<int[]> getSparseToDenseLUT () const;
    shared_ptr<Chunk[]> getOctreeSparse () const;

private:
    uint32_t getRootIndex ();
    void mergeHierarchical ();
    void initLowestOctreeHierarchy ();
    uint32_t prepareSubsampleConfig (SubsampleSet& subsampleSet, uint32_t parentIndex);
    void calculateVoxelBB (PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level);

    SubsamplingTimings randomSubsampling (
            const unique_ptr<int[]>& h_sparseToDenseLUT,
            uint32_t sparseVoxelIndex,
            uint32_t level,
            GpuArrayU32& subsampleCountingGrid,
            GpuAveraging& averagingGrid,
            GpuArrayI32& subsampleDenseToSparseLUT,
            GpuArrayU32& subsampleSparseVoxelCount,
            GpuRandomState& randomStates,
            GpuArrayU32& randomIndices);

    void evaluateOctreeProperties (
            const shared_ptr<Chunk[]>& h_octreeSparse,
            uint32_t& leafNodes,
            uint32_t& parentNodes,
            uint32_t& pointSum,
            uint32_t& min,
            uint32_t& max,
            uint32_t nodeIndex) const;

    void calculatePointVarianceInLeafNoes (
            const shared_ptr<Chunk[]>& h_octreeSparse, float& sumVariance, float& ean, uint32_t nodeIndex) const;

    void histogramBinning (
            const shared_ptr<Chunk[]>& h_octreeSparse,
            std::vector<uint32_t>& counts,
            uint32_t min,
            uint32_t binWidth,
            uint32_t nodeIndex) const;

private:
    // Point cloud
    PointCloud itsCloud;

    // Required data structures for calculation
    GpuArrayU32 itsLeafLut;
    GpuArrayU32 itsDensePointCountPerVoxel;
    GpuArrayI32 itsDenseToSparseLUT;
    GpuArrayI32 itsSparseToDenseLUT;
    GpuArrayU32 itsTmpCounting;

    // Metadata
    OctreeMetadata itsMetadata;
    SubsampleMetadata itsSubsampleMetadata;

    // Octree
    std::unique_ptr<Octree> itsOctreeData;
    std::shared_ptr<SubsamplingData> itsSubsamples;
};
