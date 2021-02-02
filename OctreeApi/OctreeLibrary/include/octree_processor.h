//
// Created by KlausP on 02.11.2020.
//

#pragma once

#include "octree_metadata.h"
#include "types.cuh"


class OctreeProcessor
{
public:
    OctreeProcessor (
            uint32_t chunkingGrid,
            uint32_t subsamplingGrid,
            uint32_t mergingThreshold,
            PointCloudMetadata cloudMetadata,
            SubsamplingStrategy strategy);

    OctreeProcessor (const OctreeProcessor&) = delete;

    void operator= (const OctreeProcessor&) = delete;

public:
    // Set point cloud
    void setPointCloudHost (uint8_t* pointCloud);
    void setPointCloudDevice (uint8_t* pointCloud);
    void setPointCloudDevice (GpuArrayU8 pointCloud);

    // Benchmarking
    void exportOctreeStatistics (const string& filePath);

    void exportHistogram (const string& filePath, uint32_t binWidth);

    // Octree pipeline
    void initialPointCounting ();

    void performCellMerging ();

    void distributePoints ();

    void performSubsampling ();

    // Calculation tools
    void calculateVoxelBB (PointCloudMetadata& metadata, uint32_t denseVoxelIndex, uint32_t level);

    // Data export
    void exportPlyNodes (const string& folderPath);

    // Debugging methods
    const OctreeMetadata& getMetadata () const;

    void updateOctreeStatistics ();

    unique_ptr<uint32_t[]> getDataLUT () const;

    unique_ptr<uint32_t[]> getDensePointCountPerVoxel () const;

    unique_ptr<int[]> getDenseToSparseLUT () const;

    unique_ptr<int[]> getSparseToDenseLUT () const;

    unique_ptr<Chunk[]> getOctreeSparse () const;

    unordered_map<uint32_t, GpuArrayU32> const& getSubsampleLUT () const;

    const std::vector<std::tuple<std::string, float>>& getTimings () const;

private:
    // Merging
    void mergeHierarchical ();

    void initLowestOctreeHierarchy ();

    // Subsampling
    SubsamplingTimings firstPointSubsampling (
            const unique_ptr<Chunk[]>& h_octreeSparse,
            const unique_ptr<int[]>& h_sparseToDenseLUT,
            uint32_t sparseVoxelIndex,
            uint32_t level,
            GpuArrayU32& subsampleCountingGrid,
            GpuArrayI32& subsampleDenseToSparseLUT,
            GpuArrayU32& subsampleSparseVoxelCount,
            GpuSubsample& subsampleConfig);

    SubsamplingTimings randomSubsampling (
            const unique_ptr<Chunk[]>& h_octreeSparse,
            const unique_ptr<int[]>& h_sparseToDenseLUT,
            uint32_t sparseVoxelIndex,
            uint32_t level,
            GpuArrayU32& subsampleCountingGrid,
            GpuArrayI32& subsampleDenseToSparseLUT,
            GpuArrayU32& subsampleSparseVoxelCount,
            GpuRandomState& randomStates,
            GpuArrayU32& randomIndices);

    void OctreeProcessor::prepareSubsampleConfig (
            SubsampleSet &subsampleSet,
            Chunk& voxel,
            const unique_ptr<Chunk[]>& h_octreeSparse,
            uint32_t& accumulatedPoints);

    float initRandomStates (unsigned int seed, GpuRandomState& states, uint32_t nodeAmount);

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
            const unique_ptr<Chunk[]>& h_octreeSparse,
            uint32_t& leafNodes,
            uint32_t& parentNodes,
            uint32_t& pointSum,
            uint32_t& min,
            uint32_t& max,
            uint32_t nodeIndex) const;

    void calculatePointVarianceInLeafNoes (
            const unique_ptr<Chunk[]>& h_octreeSparse, float& sumVariance, float& ean, uint32_t nodeIndex) const;

    void histogramBinning (
            const unique_ptr<Chunk[]>& h_octreeSparse,
            std::vector<uint32_t>& counts,
            uint32_t min,
            uint32_t binWidth,
            uint32_t nodeIndex) const;

private:
    // Point cloud
    GpuArrayU8 itsCloudData;

    // Required data structures for calculation
    GpuArrayU32 itsLeafLut;
    GpuArrayU32 itsDensePointCountPerVoxel;
    GpuArrayI32 itsDenseToSparseLUT;
    GpuArrayI32 itsSparseToDenseLUT;
    GpuOctree itsOctree;

    // Octree Metadata
    OctreeMetadata itsMetadata;

    // Pre-calculations
    vector<uint32_t> itsVoxelsPerLevel;             // Holds the voxel amount per level (dense)
    vector<uint32_t> itsGridSideLengthPerLevel;     // Holds the side length of the grid per level
    vector<uint32_t> itsLinearizedDenseVoxelOffset; // Holds the linear voxel offset for each level (dense)

    // Subsampling
    unordered_map<uint32_t, GpuArrayU32> itsParentLut;
    unordered_map<uint32_t, GpuAveraging> itsAveragingData;

    // Benchmarking
    std::vector<std::tuple<std::string, float>>
            itsTimeMeasurement; // Holds all time measurements in the form (measurementName, time)
};
