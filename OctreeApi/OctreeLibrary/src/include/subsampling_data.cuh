/***
 * @file subsampling_data.cuh
 * @author Philip Klaus
 * @brief Contains the SubsamplingData for storing subsampling-related data
 */
#pragma once
#include "types.cuh"
#include "kernel_structs.cuh"

/***
 * The class wraps subsampling-related data structures and handles the copy between host and device.
 */
class SubsamplingData
{
public:
    SubsamplingData(uint32_t estimatedPoints, uint32_t nodeAmount);
    uint32_t getPointAmount (uint32_t index);

    // ToDo: To be removed
    const std::unique_ptr<uint32_t[]>& getLutHost (uint32_t index);
    const std::unique_ptr<uint64_t[]>& getAvgHost (uint32_t index);

    void copyToHost ();

    // ----------------

    uint32_t addLinearLutEntry(uint32_t sparseIdx);
    KernelStructs::NodeOutput getNodeOutputDevice ();
    uint32_t getLinearIdx(uint32_t sparseIndex);

    OutputData *getOutputDevice();
    OutputData*  getOutputHost(uint32_t sparseIndex);

private:
    unordered_map<uint32_t, GpuArrayU32> itsLutDevice;
    unordered_map<uint32_t, GpuAveraging> itsAvgDevice;
    unordered_map<uint32_t, std::unique_ptr<uint32_t[]>> itsLutHost;
    unordered_map<uint32_t, std::unique_ptr<uint64_t[]>> itsAvgHost;

    GpuOutputData itsOutput;

    // NodeOutput
    GpuArrayU32 itsPointCounts;
    GpuArrayU32 itsPointOffsets;
    std::unique_ptr<uint32_t[]> itsPointCountsHost;
    std::unique_ptr<uint32_t[]> itsPointOffsetsHost;

    std::unique_ptr<OutputData[]> itsOutputHost;
    uint32_t itsLinearCounter;
    unordered_map<uint32_t, uint32_t> itsLinearLut; // Maps sparse indices to linear indices
};
