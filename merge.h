#ifndef MERGE_H
#define MERGE_H

#include "common.h"

// Merge parameters structure
typedef struct {
    float power_max;           // Maximum power for merge weights
    int max_frame_count;       // Maximum number of frames to merge
    float radius_max;          // Maximum radius for local statistics
    float noise_sigma;         // Noise standard deviation
    bool use_robustness;       // Whether to use robustness weights
    bool use_kernels;          // Whether to use steerable kernels
    float scale;              // Scale factor for output
    bool bayer_mode;          // Whether input is Bayer pattern
    bool iso_kernel;          // Whether to use isotropic kernel
    int tile_size;            // Size of alignment tiles
    int* cfa_pattern;         // CFA pattern for Bayer mode
    NoiseModel noise;         // Noise model parameters
    bool use_acc_rob;         // Whether to use accumulated robustness
    float max_multiplier;     // Maximum multiplier for robustness
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

void merge_image(const Image* image,
                const SteerableKernels* kernels,
                const float* robustness,
                MergeAccumulator* acc,
                const MergeParams* params);

void merge_reference(const Image* ref_image,
                    MergeAccumulator* acc,
                    const MergeParams* params);

void compute_merge_weights(const Image* image,
                         const SteerableKernels* kernels,
                         const float* robustness,
                         float* weights,
                         const MergeParams* params);

void normalize_accumulator(MergeAccumulator* acc);
Image* reconstruct_merged_image(const MergeAccumulator* acc);

#endif // MERGE_H 