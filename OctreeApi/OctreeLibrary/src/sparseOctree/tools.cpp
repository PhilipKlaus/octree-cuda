//
// Created by KlausP on 13.11.2020.
//

#include <sparseOctree.h>

void SparseOctree::calculateVoxelBB(PointCloudMetadata &metadata, uint32_t denseVoxelIndex, uint32_t level) {

    CoordinateVector<uint32_t> coords = {};

    // 1. Calculate coordinates of voxel within the actual level
    auto indexInLevel = denseVoxelIndex - itsLinearizedDenseVoxelOffset[level];
    tools::mapFromDenseIdxToDenseCoordinates(coords, indexInLevel, itsGridSideLengthPerLevel[level]);

    // 2. Calculate the bounding box for the actual voxel
    // ToDo: Include scale and offset!!!
    float side = itsMetadata.cloudMetadata.boundingBox.maximum.x - itsMetadata.cloudMetadata.boundingBox.minimum.x;
    auto cubicWidth = side / static_cast<float>(itsGridSideLengthPerLevel[level]);

    metadata.boundingBox.minimum.x = itsMetadata.cloudMetadata.boundingBox.minimum.x + coords.x * cubicWidth;
    metadata.boundingBox.minimum.y = itsMetadata.cloudMetadata.boundingBox.minimum.y + coords.y * cubicWidth;
    metadata.boundingBox.minimum.z = itsMetadata.cloudMetadata.boundingBox.minimum.z + coords.z * cubicWidth;
    metadata.boundingBox.maximum.x = metadata.boundingBox.minimum.x + cubicWidth;
    metadata.boundingBox.maximum.y = metadata.boundingBox.minimum.y + cubicWidth;
    metadata.boundingBox.maximum.z = metadata.boundingBox.minimum.z + cubicWidth;
    metadata.cloudOffset = metadata.boundingBox.minimum;
}
