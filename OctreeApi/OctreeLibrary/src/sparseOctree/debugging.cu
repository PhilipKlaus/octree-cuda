//
// Created by KlausP on 14.11.2020.
//

#include "sparseOctree.h"


template <typename coordinateType, typename colorType>
const OctreeMetadata<coordinateType>& SparseOctree<coordinateType, colorType>::getMetadata () const
{
    return itsMetadata;
}


template <typename coordinateType, typename colorType>
unique_ptr<uint32_t[]> SparseOctree<coordinateType, colorType>::getDataLUT () const
{
    return itsDataLUT->toHost ();
}


template <typename coordinateType, typename colorType>
unique_ptr<uint32_t[]> SparseOctree<coordinateType, colorType>::getDensePointCountPerVoxel () const
{
    return itsDensePointCountPerVoxel->toHost ();
}


template <typename coordinateType, typename colorType>
unique_ptr<int[]> SparseOctree<coordinateType, colorType>::getDenseToSparseLUT () const
{
    return itsDenseToSparseLUT->toHost ();
}


template <typename coordinateType, typename colorType>
unique_ptr<int[]> SparseOctree<coordinateType, colorType>::getSparseToDenseLUT () const
{
    return itsSparseToDenseLUT->toHost ();
}


template <typename coordinateType, typename colorType>
unique_ptr<Chunk[]> SparseOctree<coordinateType, colorType>::getOctreeSparse () const
{
    return itsOctree->toHost ();
}


template <typename coordinateType, typename colorType>
unordered_map<uint32_t, GpuArrayU32> const& SparseOctree<coordinateType, colorType>::
        getSubsampleLUT () const
{
    return itsSubsampleLUTs;
}


template <typename coordinateType, typename colorType>
uint32_t SparseOctree<coordinateType, colorType>::getRootIndex ()
{
    return itsMetadata.nodeAmountSparse - 1;
}

template <typename coordinateType, typename colorType>
const std::vector<std::tuple<std::string, float>>& SparseOctree<coordinateType, colorType>::getTimings () const
{
    return itsTimeMeasurement;
}



//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<float, uint8_t>
//----------------------------------------------------------------------------------------------------------------------

template const OctreeMetadata<float>& SparseOctree<float, uint8_t>::getMetadata () const;
template unique_ptr<uint32_t[]> SparseOctree<float, uint8_t>::getDataLUT () const;
template unique_ptr<uint32_t[]> SparseOctree<float, uint8_t>::getDensePointCountPerVoxel () const;
template unique_ptr<int[]> SparseOctree<float, uint8_t>::getDenseToSparseLUT () const;
template unique_ptr<int[]> SparseOctree<float, uint8_t>::getSparseToDenseLUT () const;
template unique_ptr<Chunk[]> SparseOctree<float, uint8_t>::getOctreeSparse () const;
template unordered_map<uint32_t, GpuArrayU32> const& SparseOctree<float, uint8_t>::
        getSubsampleLUT () const;
template uint32_t SparseOctree<float, uint8_t>::getRootIndex ();
template const std::vector<std::tuple<std::string, float>>& SparseOctree<float, uint8_t>::getTimings () const;


//----------------------------------------------------------------------------------------------------------------------
//                                           SparseOctree<double, uint8_t>
//----------------------------------------------------------------------------------------------------------------------

template const OctreeMetadata<double>& SparseOctree<double, uint8_t>::getMetadata () const;
template unique_ptr<uint32_t[]> SparseOctree<double, uint8_t>::getDataLUT () const;
template unique_ptr<uint32_t[]> SparseOctree<double, uint8_t>::getDensePointCountPerVoxel () const;
template unique_ptr<int[]> SparseOctree<double, uint8_t>::getDenseToSparseLUT () const;
template unique_ptr<int[]> SparseOctree<double, uint8_t>::getSparseToDenseLUT () const;
template unique_ptr<Chunk[]> SparseOctree<double, uint8_t>::getOctreeSparse () const;
template unordered_map<uint32_t, GpuArrayU32> const& SparseOctree<double, uint8_t>::
        getSubsampleLUT () const;
template uint32_t SparseOctree<double, uint8_t>::getRootIndex ();
template const std::vector<std::tuple<std::string, float>>& SparseOctree<double, uint8_t>::getTimings () const;