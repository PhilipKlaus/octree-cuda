#include "subsampling_data.cuh"

// ToDo: to be removed
void SubsamplingData::createLUT (uint32_t pointAmount, uint32_t index)
{
    itsLutDevice[index] = std::move (createGpuU32 (pointAmount, "subsampleLUT_" + to_string (index)));
}

// ToDo: to be removed
void SubsamplingData::createAvg (uint32_t pointAmount, uint32_t index)
{
    auto averaging = createGpuAveraging (pointAmount, "averagingData_" + to_string (index));
    averaging->memset (0);
    itsAvgDevice[index] = std::move (averaging);
}

// ToDo: to be removed
uint32_t* SubsamplingData::getLutDevice (uint32_t index)
{
    return itsLutDevice[index]->devicePointer ();
}

// ToDo: to be removed
uint64_t* SubsamplingData::getAvgDevice (uint32_t index)
{
    return itsAvgDevice[index]->devicePointer ();
}

uint32_t SubsamplingData::getPointAmount (uint32_t sparseIndex)
{
    return itsPointsPerSubsampleHost[getLinearIdx(sparseIndex)];
}

const std::unique_ptr<uint32_t[]>& SubsamplingData::getLutHost (uint32_t index)
{
    if (itsLutHost.find (index) == itsLutHost.end ())
    {
        itsLutHost[index] = itsLutDevice[index]->toHost ();
    }
    return itsLutHost[index];
}

const std::unique_ptr<uint64_t[]>& SubsamplingData::getAvgHost (uint32_t index)
{
    if (itsAvgHost.find (index) == itsAvgHost.end ())
    {
        itsAvgHost[index] = itsAvgDevice[index]->toHost ();
    }
    return itsAvgHost[index];
}

void SubsamplingData::copyToHost ()
{
    std::for_each (itsLutDevice.cbegin (), itsLutDevice.cend (), [&] (const auto& lutItem) {
        itsLutHost.insert (make_pair (lutItem.first, lutItem.second->toHost ()));
    });

    std::for_each (itsAvgDevice.cbegin (), itsAvgDevice.cend (), [&] (const auto& averagingItem) {
        itsAvgHost.insert (make_pair (averagingItem.first, averagingItem.second->toHost ()));
    });

    itsPointsPerSubsampleHost = itsPointsPerSubsample->toHost();
}


SubsamplingData::SubsamplingData (uint32_t estimatedPoints, uint32_t nodeAmount) : itsLinearCounter(0)
{
    itsOutput = createGpuU8(estimatedPoints * (sizeof (uint64_t) + sizeof (uint32_t)), "output");
    itsPointsPerSubsample = createGpuU32(nodeAmount, "pointCounts");
    itsPointsPerSubsample->memset (0);
}

uint32_t SubsamplingData::addLinearLutEntry (uint32_t sparseIdx)
{
    itsLinearLut[sparseIdx] = itsLinearCounter;
    return itsLinearCounter++;
}
uint32_t* SubsamplingData::getPointsPerSubsampleDevice ()
{
    return itsPointsPerSubsample->devicePointer();
}

// ToDo: to be removed
uint32_t SubsamplingData::copyPointCount (uint32_t linearIdx)
{
    return itsPointsPerSubsample->toHost()[linearIdx];
}
uint32_t SubsamplingData::getLinearIdx (uint32_t sparseIndex)
{
    return itsLinearLut[sparseIndex];
}
