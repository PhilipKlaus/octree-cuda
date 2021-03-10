/***
 * @file subsampling_data.cuh
 * @author Philip Klaus
 * @brief Contains the SubsamplingData for storing subsampling-related data
 */
#pragma once
#include "kernel_structs.cuh"
#include "types.cuh"

/***
 * The class wraps subsampling-related data structures and handles the copy between host and device.
 */
class SubsamplingData
{
public:
    SubsamplingData (uint32_t estimatedPoints, uint32_t subsamplingGrid);
    void configureNodeAmount (uint32_t nodeAmount);
    uint32_t getPointAmount (uint32_t index);

    // ToDo: To be removed
    const std::unique_ptr<uint32_t[]>& getLutHost (uint32_t index);
    const std::unique_ptr<uint64_t[]>& getAvgHost (uint32_t index);

    void copyToHost ();

    // ----------------

    uint32_t addLinearLutEntry (uint32_t sparseIdx);
    KernelStructs::NodeOutput getNodeOutputDevice ();
    uint32_t getLinearIdx (uint32_t sparseIndex);

    OutputData* getOutputDevice ();
    OutputData* getOutputHost (uint32_t sparseIndex);

    uint32_t* getCountingGrid_d ();
    uint64_t* getAverageingGrid_d ();
    int32_t* getDenseToSparseLut_d ();
    curandState_t* getRandomStates_d ();
    uint32_t* getRandomIndices_d ();
    uint32_t getGridCellAmount ();

private:
    uint32_t itsGridCellAmount;

    unordered_map<uint32_t, GpuArrayU32> itsLutDevice;
    unordered_map<uint32_t, GpuAveraging> itsAvgDevice;
    unordered_map<uint32_t, std::unique_ptr<uint32_t[]>> itsLutHost;
    unordered_map<uint32_t, std::unique_ptr<uint64_t[]>> itsAvgHost;

    // Output data
    GpuOutputData itsOutput;
    std::unique_ptr<OutputData[]> itsOutputHost;

    // NodeOutput
    GpuArrayU32 itsPointCounts;
    GpuArrayU32 itsPointOffsets;
    std::unique_ptr<uint32_t[]> itsPointCountsHost;
    std::unique_ptr<uint32_t[]> itsPointOffsetsHost;

    // Subsampling data structures
    GpuArrayU32 itsCountingGrid;
    GpuAveraging itsAveragingGrid;
    GpuArrayI32 itsDenseToSparseLut;
    GpuRandomState itsRandomStates;
    GpuArrayU32 itsRandomIndices;

    // Internal data structures
    uint32_t itsLinearCounter;
    unordered_map<uint32_t, uint32_t> itsLinearLut; // Maps sparse indices to linear indices
};
