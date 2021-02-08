//
// Created by KlausP on 14.11.2020.
//

#include "octree_processor.h"


const OctreeMetadata& OctreeProcessor::getMetadata () const
{
    return itsMetadata;
}


unique_ptr<uint32_t[]> OctreeProcessor::getDataLUT () const
{
    return itsLeafLut->toHost ();
}


unique_ptr<uint32_t[]> OctreeProcessor::getDensePointCountPerVoxel () const
{
    return itsDensePointCountPerVoxel->toHost ();
}


unique_ptr<int[]> OctreeProcessor::getDenseToSparseLUT () const
{
    return itsDenseToSparseLUT->toHost ();
}


unique_ptr<int[]> OctreeProcessor::getSparseToDenseLUT () const
{
    return itsSparseToDenseLUT->toHost ();
}


unique_ptr<Chunk[]> OctreeProcessor::getOctreeSparse () const
{
    return itsOctree->toHost ();
}


unordered_map<uint32_t, GpuArrayU32> const& OctreeProcessor::getSubsampleLUT () const
{
    return itsParentLut;
}


uint32_t OctreeProcessor::getRootIndex ()
{
    return itsMetadata.nodeAmountSparse - 1;
}


const std::vector<std::tuple<std::string, float>>& OctreeProcessor::getTimings () const
{
    return itsTimeMeasurement;
}
