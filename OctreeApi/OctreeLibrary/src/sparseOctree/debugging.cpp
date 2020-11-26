//
// Created by KlausP on 14.11.2020.
//

#include <sparseOctree.h>


const OctreeMetadata &SparseOctree::getMetadata() const {
    return itsMetadata;
}

unique_ptr<uint32_t[]> SparseOctree::getDataLUT() const {
    return itsDataLUT->toHost();
}

unique_ptr<uint32_t[]> SparseOctree::getDensePointCountPerVoxel() const {
    return itsDensePointCountPerVoxel->toHost();
}

unique_ptr<int[]> SparseOctree::getDenseToSparseLUT() const {
    return itsDenseToSparseLUT->toHost();
}

unique_ptr<int[]> SparseOctree::getSparseToDenseLUT() const {
    return itsSparseToDenseLUT->toHost();
}

unique_ptr<Chunk[]> SparseOctree::getOctreeSparse() const {
    return itsOctreeSparse->toHost();
}

unordered_map<uint32_t, unique_ptr<CudaArray<uint32_t>>> const& SparseOctree::getSubsampleLUT() const {
    return itsSubsampleLUTs;
}

uint32_t SparseOctree::getRootIndex() {
    return itsMetadata.nodeAmountSparse - 1;
}
