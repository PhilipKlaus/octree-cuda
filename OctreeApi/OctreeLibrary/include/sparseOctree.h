//
// Created by KlausP on 02.11.2020.
//

#ifndef OCTREE_LIBRARY_SPARSE_OCTREE_H
#define OCTREE_LIBRARY_SPARSE_OCTREE_H

#include <types.h>
#include <cudaArray.h>
#include <tools.cuh>

struct OctreeMetadata {

    uint32_t depth;             // The depth of the octree // ToDo: -1
    uint32_t nodeAmountDense;   // The theoretical amount of dense nodes
};

class SparseOctree {

public:

    SparseOctree(PointCloudMetadata cloudMetadata, unique_ptr<CudaArray<Vector3>> cloudData);
    SparseOctree(const SparseOctree&) = delete;
    void operator=(const SparseOctree&) = delete;

public:

    // Benchmarking
    void exportTimeMeasurements(const string &filePath);

    // Octree pipeline
    void initialPointCounting(uint32_t initialDepth);
    void performCellMerging(uint32_t threshold);
    void distributePoints();
    void performIndexing();

    // Calculation tools
    void calculateVoxelBB(BoundingBox &bb, Vector3i &coords, BoundingBox &cloud, uint32_t denseVoxelIndex, uint32_t level);
    void preCalculateOctreeParameters(uint32_t octreeDepth);

    // Data export
    void exportOctree(const string &folderPath);

    // Debugging methods
    const PointCloudMetadata& getMetadata() { return itsPointCloudMetadata; }
    unique_ptr<uint32_t[]> getDataLUT() { return itsDataLUT->toHost(); }
    unique_ptr<uint32_t[]> getDensePointCountPerVoxel();
    unique_ptr<int[]> getDenseToSparseLUT();
    unique_ptr<int[]> getSparseToDenseLUT();
    unique_ptr<Chunk[]> getOctreeSparse();
    uint32_t getVoxelAmountSparse();
    unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> const& getSubsampleLUT() const;

private:

    // Merging
    void initializeOctreeSparse(uint32_t threshold);
    void initializeBaseGridSparse();

    // Indexing
    void hierarchicalCount(const unique_ptr<Chunk[]> &h_octreeSparse,
                           const unique_ptr<int[]> &h_sparseToDenseLUT,
                           uint32_t sparseVoxelIndex,
                           uint32_t level,
                           unique_ptr<CudaArray<uint32_t>> &subsampleCountingGrid,
                           unique_ptr<CudaArray<int>> &subsampleDenseToSparseLUT,
                           unique_ptr<CudaArray<uint32_t>> &subsampleSparseVoxelCount);

    // Exporting
    uint32_t exportTreeNode(Vector3* cpuPointCloud, const unique_ptr<Chunk[]> &octreeSparse, const unique_ptr<uint32_t[]> &dataLUT, uint32_t level, uint32_t index, const string &folder);

private:

    // Point cloud
    PointCloudMetadata itsPointCloudMetadata;                             // The metadata associated with the cloud
    unique_ptr<CudaArray<Vector3>> itsCloudData;                // The cloud data

    // Required data structures for calculation
    unique_ptr<CudaArray<uint32_t>> itsDataLUT;                 // LUT for accessing point cloud data from the octree
    unique_ptr<CudaArray<uint32_t>> itsDensePointCountPerVoxel; // Holds all point counts in dense form
    unique_ptr<CudaArray<int>> itsDenseToSparseLUT;             // LUT for mapping from dense to sparse
    unique_ptr<CudaArray<int>> itsSparseToDenseLUT;             // LUT for mapping from sparse to dense
    unique_ptr<CudaArray<Chunk>> itsOctreeSparse;               // Holds the sparse octree

    // Octree Metadata
    OctreeMetadata itsMetadata;                                 // The octree metadata
    unique_ptr<CudaArray<uint32_t>> itsVoxelAmountSparse;       // Overall initial cell amount of the sparse octree

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
    unordered_map<std::string, float> itsTimeMeasurement;       // Holds all time measurements in the form (measurementName, time)

};

#endif //OCTREE_LIBRARY_SPARSE_OCTREE_H
