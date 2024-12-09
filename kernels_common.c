#include <stdlib.h>
#include "common.h"
#include "kernels.h"

SteerableKernels* create_steerable_kernels(int size, int count) {
    SteerableKernels* kernels = (SteerableKernels*)malloc(sizeof(SteerableKernels));
    if (!kernels) return NULL;

    kernels->size = size;
    kernels->count = count;
    size_t total_size = size * size * count;

    kernels->weights = (float*)malloc(total_size * sizeof(float));
    kernels->orientations = (float*)malloc(count * sizeof(float));

    if (!kernels->weights || !kernels->orientations) {
        free_steerable_kernels(kernels);
        return NULL;
    }

    return kernels;
}

void free_steerable_kernels(SteerableKernels* kernels) {
    if (kernels) {
        free(kernels->weights);
        free(kernels->orientations);
        free(kernels);
    }
} 