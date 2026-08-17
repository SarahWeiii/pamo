#pragma once
#include <cuda.h>
#include "commons.cuh"

struct RasterizeResult
{
    float * gridDist;
    bool * gridCollide;

    void free()
    {
        CHECK_CUDA(cudaFree(gridDist));
        CHECK_CUDA(cudaFree(gridCollide));
    }
};
