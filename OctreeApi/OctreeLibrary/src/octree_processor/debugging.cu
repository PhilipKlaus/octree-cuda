//
// Created by KlausP on 14.11.2020.
//

#include "octree_processpr_impl.cuh"


const OctreeMetadata& OctreeProcessorPimpl::OctreeProcessorImpl::getMetadata () const
{
    return itsMetadata;
}


unique_ptr<uint32_t[]> OctreeProcessorPimpl::OctreeProcessorImpl::getDataLUT () const
{
    return itsLeafLut->toHost ();
}


unique_ptr<uint32_t[]> OctreeProcessorPimpl::OctreeProcessorImpl::getDensePointCountPerVoxel () const
{
    return itsDensePointCountPerVoxel->toHost ();
}


unique_ptr<int[]> OctreeProcessorPimpl::OctreeProcessorImpl::getDenseToSparseLUT () const
{
    return itsDenseToSparseLUT->toHost ();
}


unique_ptr<int[]> OctreeProcessorPimpl::OctreeProcessorImpl::getSparseToDenseLUT () const
{
    return itsSparseToDenseLUT->toHost ();
}


shared_ptr<Chunk[]> OctreeProcessorPimpl::OctreeProcessorImpl::getOctreeSparse () const
{
    return itsOctreeData->getHost ();
}


unordered_map<uint32_t, GpuArrayU32> const& OctreeProcessorPimpl::OctreeProcessorImpl::getSubsampleLUT () const
{
    return itsParentLut;
}


uint32_t OctreeProcessorPimpl::OctreeProcessorImpl::getRootIndex ()
{
    return itsMetadata.nodeAmountSparse - 1;
}


const std::vector<std::tuple<std::string, float>>& OctreeProcessorPimpl::OctreeProcessorImpl::getTimings () const
{
    return itsTimeMeasurement;
}
