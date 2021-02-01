//
// Created by KlausP on 14.11.2020.
//

#include "sparseOctree.h"


const OctreeMetadata& SparseOctree::getMetadata () const
{
    return itsMetadata;
}


unique_ptr<uint32_t[]> SparseOctree::getDataLUT () const
{
    return itsDataLUT->toHost ();
}


unique_ptr<uint32_t[]> SparseOctree::getDensePointCountPerVoxel () const
{
    return itsDensePointCountPerVoxel->toHost ();
}


unique_ptr<int[]> SparseOctree::getDenseToSparseLUT () const
{
    return itsDenseToSparseLUT->toHost ();
}


unique_ptr<int[]> SparseOctree::getSparseToDenseLUT () const
{
    return itsSparseToDenseLUT->toHost ();
}



unique_ptr<Chunk[]> SparseOctree::getOctreeSparse () const
{
    return itsOctree->toHost ();
}



unordered_map<uint32_t, GpuArrayU32> const& SparseOctree::getSubsampleLUT () const
{
    return itsSubsampleLUTs;
}



uint32_t SparseOctree::getRootIndex ()
{
    return itsMetadata.nodeAmountSparse - 1;
}


const std::vector<std::tuple<std::string, float>>& SparseOctree::getTimings () const
{
    return itsTimeMeasurement;
}
