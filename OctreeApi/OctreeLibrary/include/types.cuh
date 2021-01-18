#pragma once

struct Averaging {
  float r, g, b;
  uint32_t pointCount;
};

struct SubsampleConfig {
  uint32_t *lutAdress;
  Averaging *averagingAdress;
  uint32_t  lutStartIndex;
  uint32_t pointOffsetLower;
  uint32_t pointOffsetUpper;
};


template <typename type>
using GpuArray = unique_ptr<CudaArray<type>>;

using GpuArrayU8 = GpuArray<uint8_t>;
using GpuArrayU32 = GpuArray<uint32_t>;
using GpuArrayI32 = GpuArray<int>;

using GpuOctree = GpuArray<Chunk>;
using GpuSubsample = GpuArray<SubsampleConfig>;
using GpuAveraging = GpuArray<Averaging>;
using GpuRanomState = GpuArray<curandState_t>;