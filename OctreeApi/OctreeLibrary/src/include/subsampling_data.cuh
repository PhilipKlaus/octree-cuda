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

    // ToDo: To be removed
    const std::unique_ptr<uint32_t[]>& getLutHost (uint32_t index);
    const std::unique_ptr<uint64_t[]>& getAvgHost (uint32_t index);

    void copyToHost ();

    // ----------------
    void setActiveParent(uint32_t parentNode);
    int getLastParent();

    OutputData* getOutputDevice ();
    OutputData* getOutputHost ();

    uint64_t* getAverageingGrid_d ();
    curandState_t* getRandomStates_d ();
    uint32_t* getRandomIndices_d ();
    uint32_t getGridCellAmount ();

private:
    uint32_t itsGridCellAmount;

    unordered_map<uint32_t, GpuArrayU32> itsLutDevice;
    unordered_map<uint32_t, GpuAveraging> itsAvgDevice;
    unordered_map<uint32_t, std::unique_ptr<uint32_t[]>> itsLutHost;
    unordered_map<uint32_t, std::unique_ptr<uint64_t[]>> itsAvgHost;

    // Output Info
    GpuOutputData itsOutput;
    std::unique_ptr<OutputData[]> itsOutputHost;

    // Subsampling data structures
    GpuAveraging itsAveragingGrid;
    GpuRandomState itsRandomStates;
    GpuArrayU32 itsRandomIndices;

    // Current output info
    int itsLastParent;
};
