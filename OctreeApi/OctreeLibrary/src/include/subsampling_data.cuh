/***
 * @file subsampling_data.cuh
 * @author Philip Klaus
 * @brief Contains the SubsamplingData for storing subsampling-related data
 */
#pragma once
#include "types.cuh"

/***
 * The class wraps subsampling-related data structures and handles the copy between host and device.
 */
class SubsamplingData
{
public:
    void createLUT (uint32_t pointAmount, uint32_t index);
    void createAvg (uint32_t pointAmount, uint32_t index);

    uint32_t getLutSize (uint32_t index);
    uint32_t getAvgSize (uint32_t index);

    uint32_t* getLutDevice (uint32_t index);
    Averaging* getAvgDevice (uint32_t index);

    const std::unique_ptr<uint32_t[]>& getLutHost (uint32_t index);
    const std::unique_ptr<Averaging[]>& getAvgHost (uint32_t index);

    void copyToHost ();

private:
    unordered_map<uint32_t, GpuArrayU32> itsLutDevice;
    unordered_map<uint32_t, GpuAveraging> itsAvgDevice;
    unordered_map<uint32_t, std::unique_ptr<uint32_t[]>> itsLutHost;
    unordered_map<uint32_t, std::unique_ptr<Averaging[]>> itsAvgHost;
};
