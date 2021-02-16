//
// Created by KlausP on 14.11.2020.
//

#include "octree_processor_impl.cuh"


const OctreeMetadata& OctreeProcessor::OctreeProcessorImpl::getMetadata () const
{
    return itsMetadata;
}


unique_ptr<uint32_t[]> OctreeProcessor::OctreeProcessorImpl::getDataLUT () const
{
    return itsLeafLut->toHost ();
}


unique_ptr<uint32_t[]> OctreeProcessor::OctreeProcessorImpl::getDensePointCountPerVoxel () const
{
    return itsDensePointCountPerVoxel->toHost ();
}


unique_ptr<int[]> OctreeProcessor::OctreeProcessorImpl::getDenseToSparseLUT () const
{
    return itsDenseToSparseLUT->toHost ();
}


unique_ptr<int[]> OctreeProcessor::OctreeProcessorImpl::getSparseToDenseLUT () const
{
    return itsSparseToDenseLUT->toHost ();
}


shared_ptr<Chunk[]> OctreeProcessor::OctreeProcessorImpl::getOctreeSparse () const
{
    return itsOctreeData->getHost ();
}


unordered_map<uint32_t, GpuArrayU32> const& OctreeProcessor::OctreeProcessorImpl::getSubsampleLUT () const
{
    return itsParentLut;
}


uint32_t OctreeProcessor::OctreeProcessorImpl::getRootIndex ()
{
    return itsMetadata.nodeAmountSparse - 1;
}


const std::vector<std::tuple<std::string, float>>& OctreeProcessor::OctreeProcessorImpl::getTimings () const
{
    return itsTimeMeasurement;
}
