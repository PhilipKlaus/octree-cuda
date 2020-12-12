//
// Created by KlausP on 13.11.2020.
//

#include <sparseOctree.h>

void SparseOctree::calculateVoxelBB(BoundingBox &bb, Vector3i &coords, BoundingBox &cloud, uint32_t denseVoxelIndex, uint32_t level) {

    // 1. Calculate coordinates of voxel within the actual level
    auto indexInVoxel = denseVoxelIndex - itsLinearizedDenseVoxelOffset[level];
    tools::mapFromDenseIdxToDenseCoordinates(coords, indexInVoxel, itsGridSideLengthPerLevel[level]);

    // 2. Calculate the bounding box for the actual voxel
    // ToDo: Include scale and offset!!!
    auto dimension = tools::subtract(itsMetadata.cloudMetadata.boundingBox.maximum, itsMetadata.cloudMetadata.boundingBox.minimum);
    auto cubicWidth = dimension.x / static_cast<float>(itsGridSideLengthPerLevel[level]);

    bb.minimum.x = cloud.minimum.x + coords.x * cubicWidth;
    bb.minimum.y = cloud.minimum.y + coords.y * cubicWidth;
    bb.minimum.z = cloud.minimum.z + coords.z * cubicWidth;
    bb.maximum.x = cloud.minimum.x + (coords.x + 1.f) * cubicWidth;
    bb.maximum.y = cloud.minimum.y + (coords.y + 1.f) * cubicWidth;
    bb.maximum.z = cloud.minimum.z + (coords.z + 1.f) * cubicWidth;
}
