#include "subsampling_data.cuh"

uint32_t SubsamplingData::getPointAmount (uint32_t sparseIndex)
{
    return itsPointCountsHost[getLinearIdx(sparseIndex)];
}

const std::unique_ptr<uint32_t[]>& SubsamplingData::getLutHost (uint32_t sparseIndex)
{
    if (itsLutHost.find (sparseIndex) == itsLutHost.end ())
    {
        itsLutHost[sparseIndex] = itsLutDevice[sparseIndex]->toHost ();
    }
    return itsLutHost[sparseIndex];
}

const std::unique_ptr<uint64_t[]>& SubsamplingData::getAvgHost (uint32_t sparseIndex)
{
    if (itsAvgHost.find (sparseIndex) == itsAvgHost.end ())
    {
        itsAvgHost[sparseIndex] = itsAvgDevice[sparseIndex]->toHost ();
    }
    return itsAvgHost[sparseIndex];
}

void SubsamplingData::copyToHost ()
{
    itsPointCountsHost = itsPointCounts->toHost();
    itsPointOffsetsHost = itsPointOffsets->toHost();
    itsOutputHost     = itsOutput->toHost();
}


SubsamplingData::SubsamplingData (uint32_t estimatedPoints, uint32_t nodeAmount) : itsLinearCounter(0)
{
    itsOutput = createGpuOutputData(estimatedPoints, "output");
    itsOutput->memset(0);

    itsPointCounts = createGpuU32(nodeAmount, "pointCounts");
    itsPointOffsets = createGpuU32(nodeAmount, "pointOffsets");
    itsPointCounts->memset(0);
}

uint32_t SubsamplingData::addLinearLutEntry (uint32_t sparseIdx)
{
    itsLinearLut[sparseIdx] = itsLinearCounter;
    return itsLinearCounter++;
}
KernelStructs::NodeOutput SubsamplingData::getNodeOutputDevice ()
{
    return {itsPointCounts->devicePointer(), itsPointOffsets->devicePointer() };
}


uint32_t SubsamplingData::getLinearIdx (uint32_t sparseIndex)
{
    return itsLinearLut[sparseIndex];
}
OutputData* SubsamplingData::getOutputDevice ()
{
    return itsOutput->devicePointer();
}
OutputData* SubsamplingData::getOutputHost (uint32_t sparseIndex)
{
    uint64_t byteOffset = itsPointOffsetsHost[getLinearIdx(sparseIndex)];
    return (itsOutputHost.get() + byteOffset);
}
