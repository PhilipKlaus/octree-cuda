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
    auto width = dimension.x / itsGridSideLengthPerLevel[level];
    auto height = dimension.y / itsGridSideLengthPerLevel[level];
    auto depth = dimension.z / itsGridSideLengthPerLevel[level];

    bb.minimum.x = cloud.minimum.x + coords.x * width;
    bb.minimum.y = cloud.minimum.y + coords.y * height;
    bb.minimum.z = cloud.minimum.z + coords.z * depth;
    bb.maximum.x = cloud.minimum.x + (coords.x + 1.f) * width;
    bb.maximum.y = cloud.minimum.y + (coords.y + 1.f) * height;
    bb.maximum.z = cloud.minimum.z + (coords.z + 1.f) * depth;
}
