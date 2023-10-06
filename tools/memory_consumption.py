"""
This Python scripts helps to estimate the GPU memory consumption for
PotreeConverterGPU. The scripts does not consider data structures which
depend on actual sparse node amount as this depends heavily on the
spatial point distribution in the point cloud.
Thus, the purpose of this script is to estimate the minimum required GPU
memory amount and vice versa the maximum processable points for a given configuration
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from math import log, pow
from matplotlib.ticker import FormatStrFormatter


@dataclass
class ProcessingCfg:
    chunkingGridSize = 512
    subsamplingGridSize = 128
    outputFactor = 2.2


@dataclass
class Gpu:
    name = "NVIDIA TITAN RTX"
    memory_gb = 24576000000 / 1000000000


def calculate_memory_function(cfg: ProcessingCfg, gpu: Gpu, inputDataStride: float):
    # Define constants
    GB = 1000000000
    CURAND_STATE_SIZE = 48 / GB
    OUTPUT_DATA_STRIDE = 18 / GB

    levels = int(log(cfg.chunkingGridSize, 10) / log(2, 10))
    countingGrid = sum([pow(512 >> i, 3) * (4 / GB) for i in range(levels + 1)])
    denseToSparseLUT = countingGrid
    tmpCounting = 4 / GB
    randomIndices = (4 / GB) * pow(cfg.subsamplingGridSize, 3)
    randomStates = CURAND_STATE_SIZE * 1024
    averagingGrid = (8 / GB) * pow(cfg.subsamplingGridSize, 3)

    # Define lambdas for gpu data structures
    outputBuffer = lambda points: points * OUTPUT_DATA_STRIDE * cfg.outputFactor
    pointCloud = lambda points: points * inputDataStride
    pointLut = lambda points: points * cfg.outputFactor * (4 / GB)

    # linearly spaced point amounts
    x = np.linspace(0, 400000000, 1000)

    """
    Equation:
    mem = output + cloud + countingGrid + denseToSpare + tmpCounting + pointLut + randomIndices + randomStates + avgGrid
    mem - countingGrid - denseToSparse - tmpCounting - randomIndices - randomStates - avgGrid = output + cloud + pointLUT
    (...) = output + cloud + pointLUT
    (...) = points * (outputStride * outputFactor) + points * inputDataStride + points * (outputFactor * (4 / GB))
    (...) = points * (outputStride * outputFactor + inputDataStride + outputFactor * (4 / GB))
    (...) / (outputStride * outputFactor + inputDataStride + outputFactor * (4 / GB)) = points
    (...) / (outputFactor * (outputStride + inputDataStride + (4 / GB)))
    """
    estimated_points = \
        (gpu.memory_gb - countingGrid - denseToSparseLUT - tmpCounting - randomIndices - randomStates - averagingGrid) / \
        (cfg.outputFactor * (OUTPUT_DATA_STRIDE + (4 / GB)) + inputDataStride)

    # the function of the point amount
    return x, outputBuffer(x) + \
           pointCloud(x) + \
           countingGrid + \
           denseToSparseLUT + \
           tmpCounting + \
           pointLut(x) + \
           randomIndices + \
           randomStates + \
           averagingGrid, estimated_points


def draw_estimation(x, y, estimated, gpu: Gpu, color='r'):
    plt.plot(x, y, color)

    plt.plot([0, estimated], [gpu.memory_gb, gpu.memory_gb], color='k', linestyle='--', linewidth=1)
    plt.plot([estimated, estimated], [0, gpu.memory_gb], color='k', linestyle='--', linewidth=1)
    plt.plot(estimated, gpu.memory_gb, 'b*', linewidth=3)


if __name__ == "__main__":
    gpu = Gpu()
    cfg = ProcessingCfg()

    plt.ylabel('minimum memory consumption [GB]')
    plt.xlabel('maximum amount of points [10^6 points]')

    # Estimate for double precision coordinates
    x, y, estimated_double = calculate_memory_function(cfg, gpu, 27 / 1000000000)
    draw_estimation(x / 1000000, y, estimated_double / 1000000, gpu)

    # Estimate for single precision coordinates
    x, y, estimated_single = calculate_memory_function(cfg, gpu, 15 / 1000000000)
    draw_estimation(x / 1000000, y, estimated_single / 1000000, gpu, 'g')

    print(f"Summary:\nmax points (double prec.): {estimated_double}\nmax points (single prec.): {estimated_single}")
    plt.legend(['double precision coordinates', 'single precision coordinates'], loc=0)
    # plt.legend(['double precision coordinates', 'single precision coordinates'], loc=0)
    plt.annotate(gpu.name, (0, gpu.memory_gb + 1))

    # plt.axes().xaxis.set_major_formatter(FormatStrFormatter('%.0fM'))

    # save and show the plot
    plt.savefig("gpu_memory_estimation.pdf")
    plt.show()
