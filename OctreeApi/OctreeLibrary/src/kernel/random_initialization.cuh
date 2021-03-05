#include "kernel_executor.cuh"

namespace subsampling {

/**
 * Initializes a CUDA random state.
 * @param seed The actual seed for the randomization.
 * @param states The CUDA random states which shoul be initialized.
 * @param cellAmount The amount of cells for which the kernel is called.
 */
__global__ void kernelInitRandoms (unsigned int seed, curandState_t* states, uint32_t cellAmount)
{
    int index = (blockIdx.y * gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= cellAmount)
    {
        return;
    }

    curand_init (seed, index, 0, &states[index]);
}
} // namespace subsampling