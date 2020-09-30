#pragma once
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include "../../Core/src/default.cuh"
#include "pnm-io/pnm_io.h"

namespace Core {
namespace Tests {



template <typename T>
void PrintData (Core::TensorView<T, 2> data, int precision)
{
    Core::gpuErrchk (cudaDeviceSynchronize ());

    std::stringstream stream;
    stream << std::fixed << std::setprecision (precision);

    for (int i = 0; i < data.Shape (1); ++i)
    {
        stream << "\"";
        for (int j = 0; j < data.Shape (0); ++j)
        {
            stream << (float)data (j, i) << " ";
        }
        stream << "\"\n";
    }

    std::cout << stream.str ().c_str () << std::endl;
}
} // end namespace Tests
} // end namespace Core
