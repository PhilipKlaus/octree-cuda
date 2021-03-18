#include "random_initialization.cuh"
#include "subsampling_data.cuh"


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

SubsamplingData::SubsamplingData (uint32_t subsamplingGrid) : itsLastParent(-1)
{
    itsGridCellAmount   = static_cast<uint32_t> (pow (subsamplingGrid, 3.f));
    itsAveragingGrid    = createGpuAveraging (itsGridCellAmount, "averagingGrid");
    itsRandomStates     = createGpuRandom (1024, "randomStates");
    itsRandomIndices    = createGpuU32 (itsGridCellAmount, "randomIndices");

    itsAveragingGrid->memset(0);

    executeKernel (
            subsampling::kernelInitRandoms,
            1024u,
            "kernelInitRandoms",
            std::time (nullptr),
            itsRandomStates->devicePointer (),
            1024);
}

uint64_t* SubsamplingData::getAverageingGrid_d ()
{
    return itsAveragingGrid->devicePointer ();
}

curandState_t* SubsamplingData::getRandomStates_d ()
{
    return itsRandomStates->devicePointer ();
}
uint32_t* SubsamplingData::getRandomIndices_d ()
{
    return itsRandomIndices->devicePointer ();
}

uint32_t SubsamplingData::getGridCellAmount ()
{
    return itsGridCellAmount;
}

void SubsamplingData::setActiveParent (uint32_t parentNode)
{
    itsLastParent = static_cast<int>(parentNode);
}
int SubsamplingData::getLastParent ()
{
    return itsLastParent;
}
