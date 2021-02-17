#include "octree_processor.cuh"
#include "octree_processor_impl.cuh"

OctreeProcessor::OctreeProcessor (
        uint8_t* pointCloud,
        uint32_t chunkingGrid,
        uint32_t mergingThreshold,
        PointCloudMetadata cloudMetadata,
        SubsampleMetadata subsamplingMetadata)
{
    itsProcessor = std::make_unique<OctreeProcessor::OctreeProcessorImpl> (
            pointCloud, chunkingGrid, mergingThreshold, cloudMetadata, subsamplingMetadata);
}

void OctreeProcessor::initialPointCounting ()
{
    itsProcessor->initialPointCounting ();
}

void OctreeProcessor::performCellMerging ()
{
    itsProcessor->performCellMerging ();
}
void OctreeProcessor::distributePoints ()
{
    itsProcessor->distributePoints ();
}
void OctreeProcessor::performSubsampling ()
{
    itsProcessor->performSubsampling ();
}

void OctreeProcessor::exportPlyNodes (const std::string& folderPath)
{
    itsProcessor->exportPlyNodes (folderPath);
}

void OctreeProcessor::exportPotree (const std::string& folderPath)
{
    itsProcessor->exportPotree (folderPath);
}

void OctreeProcessor::updateStatistics ()
{
    itsProcessor->updateOctreeStatistics ();
}

void OctreeProcessor::exportHistogram (const std::string& filePath, uint32_t binWidth)
{
    itsProcessor->exportHistogram (filePath, binWidth);
}

const OctreeMetadata& OctreeProcessor::getOctreeMetadata ()
{
    return itsProcessor->getMetadata ();
}

OctreeProcessor::~OctreeProcessor ()
{
    // Empty destructor - necessary becaus of PIMPL
    // https://stackoverflow.com/questions/9954518/stdunique-ptr-with-an-incomplete-type-wont-compile
}