//
// Created by KlausP on 13.11.2020.
//

#include <sparseOctree.h>

void SparseOctree::calculateVoxelBB(BoundingBox &bb, CoordinateVector<uint32_t> &coords, uint32_t denseVoxelIndex, uint32_t level) {

    // 1. Calculate coordinates of voxel within the actual level
    auto indexInLevel = denseVoxelIndex - itsLinearizedDenseVoxelOffset[level];
    tools::mapFromDenseIdxToDenseCoordinates(coords, indexInLevel, itsGridSideLengthPerLevel[level]);

    // 2. Calculate the bounding box for the actual voxel
    // ToDo: Include scale and offset!!!
    float side = itsMetadata.cloudMetadata.boundingBox.maximum.x - itsMetadata.cloudMetadata.boundingBox.minimum.x;
    auto cubicWidth = side / static_cast<float>(itsGridSideLengthPerLevel[level]);

    bb.minimum.x = itsMetadata.cloudMetadata.boundingBox.minimum.x + coords.x * cubicWidth;
    bb.minimum.y = itsMetadata.cloudMetadata.boundingBox.minimum.y + coords.y * cubicWidth;
    bb.minimum.z = itsMetadata.cloudMetadata.boundingBox.minimum.z + coords.z * cubicWidth;
    bb.maximum.x = bb.minimum.x + cubicWidth;
    bb.maximum.y = bb.minimum.y + cubicWidth;
    bb.maximum.z = bb.minimum.z + cubicWidth;
}
