//
// Created by KlausP on 29.01.2021.
//

#pragma once

template <typename coordinateType>
std::vector<coordinateType> calculateRealBB (const uint8_t* cloud, uint32_t pointAmount, uint32_t dataStride)
{
    std::vector<coordinateType> bbReal = {
            INFINITY,
            INFINITY,
            INFINITY,
            -INFINITY,
            -INFINITY,
            -INFINITY
    };

    uint8_t positionSize = sizeof (coordinateType);

    for (auto i = 0; i < pointAmount; ++i)
    {
        coordinateType pointX = *reinterpret_cast<const coordinateType*> (cloud + i * dataStride);
        coordinateType pointY = *reinterpret_cast<const coordinateType*> (cloud + i * dataStride + positionSize);
        coordinateType pointZ = *reinterpret_cast<const coordinateType*> (cloud + i * dataStride + positionSize * 2);

        bbReal[0] = fmin (bbReal[0], pointX);
        bbReal[1] = fmin (bbReal[1], pointY);
        bbReal[2] = fmin (bbReal[2], pointZ);
        bbReal[3] = fmax (bbReal[3], pointX);
        bbReal[4] = fmax (bbReal[4], pointY);
        bbReal[5] = fmax (bbReal[5], pointZ);
    }

    return bbReal;
}

template <typename coordinateType>
std::vector<coordinateType> calculateCubicBB (const std::vector<coordinateType> &realBB)
{
    auto dimX = realBB[3] - realBB[0];
    auto dimY = realBB[4] - realBB[1];
    auto dimZ = realBB[5] - realBB[2];

    coordinateType cubicSideLength = max (max (dimX, dimY), dimZ);

    std::vector<coordinateType> cubicBB;

    cubicBB.push_back(realBB[0] - ((cubicSideLength - dimX) / 2.0f));
    cubicBB.push_back(realBB[1] - ((cubicSideLength - dimY) / 2.0f));
    cubicBB.push_back(realBB[2] - ((cubicSideLength - dimZ) / 2.0f));
    cubicBB.push_back(cubicBB[0] + cubicSideLength);
    cubicBB.push_back(cubicBB[1] + cubicSideLength);
    cubicBB.push_back(cubicBB[2] + cubicSideLength);

    return cubicBB;
}