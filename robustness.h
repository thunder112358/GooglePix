#ifndef ROBUSTNESS_H
#define ROBUSTNESS_H

#include <stdbool.h>
#include <math.h>
#include "utils.h"
#include "block_matching.h"
#include "linalg.h"  // For Matrix2x2 and bilinear_interpolate

// Robustness parameters structure
typedef struct {
    bool enabled;           // Whether robustness is activated
    float threshold;        // Threshold parameter t
    float s1;              // Scale parameter s1
    float s2;              // Scale parameter s2
    int mt;                // Mt parameter
    int window_size;       // Size of local statistics window
    float epsilon;         // Small value to prevent division by zero
    NoiseModel noise;      // Noise model parameters
    bool bayer_mode;       // Whether input is Bayer pattern
    int* cfa_pattern;      // CFA pattern for Bayer mode
} RobustnessParams;

// Local statistics structure
typedef struct {
    float* means;          // Local mean values
    float* stds;           // Local standard deviation values
    int height;            // Image height
    int width;             // Image width
    int channels;          // Number of channels
} LocalStats;

// Add noise model structure
typedef struct {
    float* std_curve;     // Noise model for sigma
    float* diff_curve;    // Noise model for d
    int curve_size;       // Size of noise curves (typically 1000)
} NoiseModel;

// Add guide image structure
typedef struct {
    float* data;
    int height;
    int width;
    int channels;
} GuideImage;

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

// Add function declarations
GuideImage* compute_guide_image(const Image* raw_img, const int* cfa_pattern);
void free_guide_image(GuideImage* guide);
void compute_local_min_5x5(const float* input, float* output, int height, int width);
void apply_noise_model(const float* d_p, const float* ref_means, const float* ref_stds,
                      const NoiseModel* noise, float* d_sq, float* sigma_sq,
                      int height, int width, int channels);
void compute_flow_irregularity(const AlignmentMap* flow, const RobustnessParams* params,
                             float* S);

#endif // ROBUSTNESS_H 