#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdbool.h>

// Common type definitions
typedef float pixel_t;  // Default float type for pixel values

// Common structures
typedef struct {
    float* data;
    int height;
    int width;
    int channels;
} Image;

typedef struct {
    float a11, a12;
    float a21, a22;
} Matrix2x2;

typedef struct {
    float* std_curve;
    float* diff_curve;
    int curve_size;
} NoiseModel;

// Common constants
#define MAX_PYRAMID_LEVELS 10

#endif // COMMON_H 