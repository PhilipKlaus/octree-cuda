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
    SubsamplingData (uint32_t subsamplingGrid);

    // ToDo: To be removed
    const std::unique_ptr<uint32_t[]>& getLutHost (uint32_t index);
    const std::unique_ptr<uint64_t[]>& getAvgHost (uint32_t index);

    // ----------------
    void setActiveParent (uint32_t parentNode);
    int getLastParent ();

    uint32_t getGridCellAmount ();

private:
    uint32_t itsGridCellAmount;

    unordered_map<uint32_t, GpuArrayU32> itsLutDevice;
    unordered_map<uint32_t, GpuAveraging> itsAvgDevice;
    unordered_map<uint32_t, std::unique_ptr<uint32_t[]>> itsLutHost;
    unordered_map<uint32_t, std::unique_ptr<uint64_t[]>> itsAvgHost;

    // Current output info
    int itsLastParent;
};
