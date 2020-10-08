//
// Created by KlausP on 30.09.2020.
//

#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

using namespace std;

struct Vector3
{
    float x, y, z;
};

struct BoundingBox {
    Vector3 minimum;
    Vector3 maximum;

    Vector3 size() const {
        return {
            maximum.x - minimum.x,
            maximum.y - minimum.y,
            maximum.z - minimum.z
        };
    }
};

struct PointCloudMetadata {
    uint32_t pointAmount;
    BoundingBox boundingBox;
    Vector3 cloudOffset;
};

template <typename dataType>
class CudaArray {

public:

    CudaArray(unsigned int elements) : itsElements(elements) {
        cudaMalloc((void**)&itsData, itsElements * sizeof(dataType));
        cout << "Reserved memory on GPU" << " for " << elements << " pointCount ["
             << sizeof(dataType) * elements << " bytes] " << endl;
    }

    ~CudaArray() {
        cudaFree(itsData);
        cout << "Freed memory on GPU" << " from " << itsElements << " pointCount ["
             << sizeof(dataType) * itsElements << " bytes] " << endl;
    }

    dataType* devicePointer() {
        return itsData;
    }

    unique_ptr<dataType[]> toHost() {
        unique_ptr<dataType[]> host (new dataType[itsElements]);
        cudaMemcpy(host.get(), itsData, sizeof(dataType) * itsElements, cudaMemcpyDeviceToHost);
        return host;
    }

    void toGPU(uint8_t *host) {
        cudaMemcpy(itsData, host, sizeof(dataType) * itsElements, cudaMemcpyHostToDevice);
    }

    uint32_t pointCount() {
        return itsElements;
    }

    unsigned int itsElements;
    dataType *itsData;
};
