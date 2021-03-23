#include "octree_processor.cuh"
#include "octree_processor_impl.cuh"

OctreeProcessor::OctreeProcessor (uint8_t* pointCloud, PointCloudInfo cloudInfo, ProcessingInfo processingInfo)
{
    itsProcessor = std::make_unique<OctreeProcessor::OctreeProcessorImpl> (pointCloud, cloudInfo, processingInfo);
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
    itsProcessor->updateOctreeInfo ();
}

void OctreeProcessor::exportHistogram (const std::string& filePath, uint32_t binWidth)
{
    itsProcessor->exportHistogram (filePath, binWidth);
}

const OctreeInfo& OctreeProcessor::getNodeStatistics ()
{
    return itsProcessor->getOctreeInfo ();
}

OctreeProcessor::~OctreeProcessor ()
{
    // Empty destructor - necessary becaus of PIMPL
    // https://stackoverflow.com/questions/9954518/stdunique-ptr-with-an-incomplete-type-wont-compile
}