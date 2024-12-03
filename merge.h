#ifndef MERGE_H
#define MERGE_H

#include "utils.h"
#include "kernels.h"
#include "robustness.h"

// Merge parameters structure
typedef struct {
    float power_max;           // Maximum power for merge weights
    int max_frame_count;       // Maximum number of frames to merge
    float radius_max;          // Maximum radius for local statistics
    float noise_sigma;         // Noise standard deviation
    bool use_robustness;       // Whether to use robustness weights
    bool use_kernels;          // Whether to use steerable kernels
} MergeParams;

// Merge accumulator structure
typedef struct {
    float* numerator;          // Accumulated weighted sum
    float* denominator;        // Accumulated weights
    int height;               // Image height
    int width;                // Image width
    int channels;             // Number of channels
} MergeAccumulator;

// Function declarations
MergeAccumulator* create_merge_accumulator(int height, int width, int channels);
void free_merge_accumulator(MergeAccumulator* acc);

// Main merge functions
void merge_image(const Image* image,
                const SteerableKernels* kernels,
                const float* robustness,
                MergeAccumulator* acc,
                const MergeParams* params);

void merge_reference(const Image* ref_image,
                    MergeAccumulator* acc,
                    const MergeParams* params);

// Helper functions
void compute_merge_weights(const Image* image,
                         const SteerableKernels* kernels,
                         const float* robustness,
                         float* weights,
                         const MergeParams* params);

void normalize_accumulator(MergeAccumulator* acc);

// Final image reconstruction
Image* reconstruct_merged_image(const MergeAccumulator* acc);

#endif // MERGE_H 