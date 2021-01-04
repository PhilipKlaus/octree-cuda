//
// Created by KlausP on 02.11.2020.
//

#ifndef OCTREE_LIBRARY_SPARSE_OCTREE_H
#define OCTREE_LIBRARY_SPARSE_OCTREE_H

#include <types.h>
#include <cudaArray.h>
#include <tools.cuh>


class SparseOctree {

public:

    SparseOctree(
            GridSize chunkingGrid,
            GridSize subsamplingGrid,
            uint32_t mergingThreshold,
            PointCloudMetadata cloudMetadata,
            unique_ptr<CudaArray<uint8_t>> cloudData,
            SubsamplingStrategy strategy);

    SparseOctree(const SparseOctree&) = delete;
    void operator=(const SparseOctree&) = delete;

public:

    // Benchmarking
    void exportOctreeStatistics(const string &filePath);
    void exportHistogram(const string &filePath, uint32_t binWidth);

    // Octree pipeline
    void initialPointCounting();
    void performCellMerging();
    void distributePoints();
    void performSubsampling();

    // Calculation tools
    void calculateVoxelBB(BoundingBox &bb, CoordinateVector<uint32_t> &coords, uint32_t denseVoxelIndex, uint32_t level);

    // Data export
    void exportPlyNodes(const string &folderPath);

    // Debugging methods
    const OctreeMetadata& getMetadata() const;
    unique_ptr<uint32_t[]> getDataLUT()const;
    unique_ptr<uint32_t[]> getDensePointCountPerVoxel() const;
    unique_ptr<int[]> getDenseToSparseLUT() const;
    unique_ptr<int[]> getSparseToDenseLUT() const;
    unique_ptr<Chunk[]> getOctreeSparse() const;
    unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> const& getSubsampleLUT() const;

private:

    // Merging
    void mergeHierarchical();
    void initLowestOctreeHierarchy();

    // Subsampling
    float hierarchicalSubsampling(const unique_ptr<Chunk[]> &h_octreeSparse,
                                 const unique_ptr<int[]> &h_sparseToDenseLUT,
                                 uint32_t sparseVoxelIndex,
                                 uint32_t level,
                                 unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
                                 unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
                                 unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount);

    // Exporting
    uint32_t exportTreeNode(uint8_t* cpuPointCloud, const unique_ptr<Chunk[]> &octreeSparse, const unique_ptr<uint32_t[]> &dataLUT, const string& level, uint32_t index, const string &folder);

    // Benchmarking
    uint32_t getRootIndex();
    void updateOctreeStatistics();
    void evaluateOctreeProperties(const unique_ptr<Chunk[]> &h_octreeSparse, uint32_t &leafNodes, uint32_t &parentNodes, uint32_t &pointSum, uint32_t  &min, uint32_t &max, uint32_t nodeIndex) const;
    void calculatePointVarianceInLeafNoes(const unique_ptr<Chunk[]> &h_octreeSparse, float &sumVariance, float &ean, uint32_t nodeIndex) const;
    void histogramBinning(const unique_ptr<Chunk[]> &h_octreeSparse, std::vector<uint32_t> &counts, uint32_t min, uint32_t binWidth, uint32_t nodeIndex) const;

private:

    // Point cloud
    unique_ptr<CudaArray<uint8_t>> itsCloudData;                // The cloud data

    // Required data structures for calculation
    unique_ptr<CudaArray<uint32_t>> itsDataLUT;                 // LUT for accessing point cloud data from the octree
    unique_ptr<CudaArray<uint32_t>> itsDensePointCountPerVoxel; // Holds all point counts in dense form
    unique_ptr<CudaArray<int>> itsDenseToSparseLUT;             // LUT for mapping from dense to sparse
    unique_ptr<CudaArray<int>> itsSparseToDenseLUT;             // LUT for mapping from sparse to dense
    unique_ptr<CudaArray<Chunk>> itsOctreeSparse;               // Holds the sparse octree

    // Octree Metadata
    OctreeMetadata itsMetadata;                                 // The octree metadata

    // Pre-calculations
    vector<uint32_t> itsVoxelsPerLevel ;                        // Holds the voxel amount per level (dense)
    vector<uint32_t> itsGridSideLengthPerLevel;                 // Holds the side length of the grid per level
                                                                // E.g.: level 0 -> 128x128x128 -> side length: 128
    vector<uint32_t> itsLinearizedDenseVoxelOffset;             // Holds the linear voxel offset for each level (dense)
                                                                // Level 0 is e.g. 128x128x128
                                                                // Offset for level 0 = 0
                                                                // Offset for level 1 = level 0 + 128 x 128 x128

    // Subsampling
    unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> itsSubsampleLUTs;

    // Benchmarking
    std::vector<std::tuple<std::string, float>> itsTimeMeasurement; // Holds all time measurements in the form (measurementName, time)

};

#endif //OCTREE_LIBRARY_SPARSE_OCTREE_H
