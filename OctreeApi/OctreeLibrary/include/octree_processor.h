//
// Created by KlausP on 02.11.2020.
//

#pragma once

#include "octree_data.cuh"
#include "octree_metadata.h"
#include "point_cloud.cuh"
#include "types.cuh"

class OctreeProcessor
{
public:
    OctreeProcessor (
            uint8_t* pointCloud,
            uint32_t chunkingGrid,
            uint32_t mergingThreshold,
            PointCloudMetadata cloudMetadata,
            SubsampleMetadata subsamplingMetadata);

    OctreeProcessor (const OctreeProcessor&) = delete;

    void operator= (const OctreeProcessor&) = delete;

public:
    ///@{
    ///@name Chunking
    void initialPointCounting ();
    void performCellMerging ();
    void distributePoints ();
    void performSubsampling ();
    ///@}

    const OctreeMetadata& getMetadata () const;

    void exportHistogram (const string& filePath, uint32_t binWidth);
    void exportPlyNodes (const string& folderPath);
    void updateOctreeStatistics ();
    unique_ptr<uint32_t[]> getDataLUT () const;
    unique_ptr<uint32_t[]> getDensePointCountPerVoxel () const;
    unique_ptr<int[]> getDenseToSparseLUT () const;
    unique_ptr<int[]> getSparseToDenseLUT () const;
    shared_ptr<Chunk[]> getOctreeSparse () const;
    unordered_map<uint32_t, GpuArrayU32> const& getSubsampleLUT () const;
    const std::vector<std::tuple<std::string, float>>& getTimings () const;

private:
    // Merging
    void mergeHierarchical ();

    void initLowestOctreeHierarchy ();

    // Subsampling
    SubsamplingTimings randomSubsampling (
            const shared_ptr<Chunk[]>& h_octreeSparse,
            const unique_ptr<int[]>& h_sparseToDenseLUT,
            uint32_t sparseVoxelIndex,
            uint32_t level,
            GpuArrayU32& subsampleCountingGrid,
            GpuAveraging& averagingGrid,
            GpuArrayI32& subsampleDenseToSparseLUT,
            GpuArrayU32& subsampleSparseVoxelCount,
            GpuRandomState& randomStates,
            GpuArrayU32& randomIndices);

    uint32_t OctreeProcessor::prepareSubsampleConfig (
            SubsampleSet& subsampleSet, Chunk& voxel, const shared_ptr<Chunk[]>& h_octreeSparse);

    // Exporting
    uint32_t exportTreeNode (
            uint8_t* cpuPointCloud,
            const unique_ptr<Chunk[]>& octreeSparse,
            const unique_ptr<uint32_t[]>& dataLUT,
            const string& level,
            uint32_t index,
            const string& folder);

    // Benchmarking
    uint32_t getRootIndex ();

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

    void calculateVoxelBB (PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level);

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
    std::unique_ptr<OctreeData> itsOctreeData;

    // Subsampling
    unordered_map<uint32_t, GpuArrayU32> itsParentLut;
    unordered_map<uint32_t, GpuAveraging> itsAveragingData;

    // Benchmarking
    std::vector<std::tuple<std::string, float>>
            itsTimeMeasurement; // Holds all time measurements in the form (measurementName, time)
};
