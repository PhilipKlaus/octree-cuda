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
    OctreeProcessorImpl (uint8_t* pointCloud, PointCloudInfo cloudMetadata, ProcessingInfo subsamplingMetadata);
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

    const OctreeInfo& getOctreeInfo ();
    void updateOctreeInfo ();

private:
    void mergeHierarchical ();
    void initLowestOctreeHierarchy ();
    void calculateVoxelBB (PointCloudInfo& metadata, uint32_t denseVoxelIndex, uint32_t level);
    void randomSubsampling (uint32_t sparseVoxelIndex, uint32_t level, Vector3<double> nodeBBMin);
    void firstPointSubsampling (uint32_t sparseVoxelIndex, uint32_t level, Vector3<double> nodeBBMin);

    void histogramBinning (std::vector<uint32_t>& counts, uint32_t min, uint32_t binWidth, uint32_t nodeIndex) const;

    void setActiveParent (uint32_t parentNode);
    int getLastParent () const;


private:
    PointCloud itsCloud;
    Octree itsOctree;

    ProcessingInfo itsProcessingInfo;

    // GPU helper data
    GpuArrayU32 itsCountingGrid;
    GpuArrayI32 itsDenseToSparseLUT;
    GpuArrayU32 itsTmpCounting;
    GpuPointLut itsPointLut;
    GpuAveraging itsAveragingGrid;
    GpuRandomState itsRandomStates;
    GpuArrayU32 itsRandomIndices;
    int itsLastSubsampleNode;
};
