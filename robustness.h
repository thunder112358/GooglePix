#ifndef ROBUSTNESS_H
#define ROBUSTNESS_H

#include <stdbool.h>
#include "utils.h"
#include "block_matching.h"

// Robustness parameters structure
typedef struct {
    bool enabled;           // Whether robustness is activated
    float threshold;        // Threshold parameter t
    float s1;              // Scale parameter s1
    float s2;              // Scale parameter s2
    int mt;                // Mt parameter
    int window_size;       // Size of local statistics window
    float epsilon;         // Small value to prevent division by zero
} RobustnessParams;

// Local statistics structure
typedef struct {
    float* means;          // Local mean values
    float* stds;           // Local standard deviation values
    int height;            // Image height
    int width;             // Image width
    int channels;          // Number of channels
} LocalStats;

// Function declarations
LocalStats* init_robustness(const Image* image, const RobustnessParams* params);
void free_local_stats(LocalStats* stats);

// Compute robustness map for an image
float* compute_robustness(const Image* image, 
                         const LocalStats* ref_stats,
                         const AlignmentMap* alignment,
                         const RobustnessParams* params);

// Helper functions
float* compute_guide_image(const Image* image, const RobustnessParams* params);
float* compute_local_min(const float* values, int height, int width, int radius);
void compute_local_statistics(const Image* image, 
                            const RobustnessParams* params,
                            float* means, 
                            float* stds);

// Utility functions
float dogson_biquadratic_kernel(float x, float y);
float dogson_quadratic_kernel(float x);

#endif // ROBUSTNESS_H 