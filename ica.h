/**
 * @file ica.h
 * @brief Iterative Closest Algorithm (ICA) implementation for image alignment refinement
 */

#ifndef ICA_H
#define ICA_H

#include "block_matching.h"

// Structures for ICA
typedef struct {
    pixel_t* data_x;  // Horizontal gradients
    pixel_t* data_y;  // Vertical gradients
    int height;
    int width;
} ImageGradients;

typedef struct {
    float* data;      // 2x2 matrices stored in row-major order
    int height;       // Number of patches in y direction
    int width;        // Number of patches in x direction
} HessianMatrix;

// Parameters structure for ICA
typedef struct {
    float sigma_blur;     // Gaussian blur sigma (0 means no blur)
    int num_iterations;   // Number of Kanade iterations
    int tile_size;       // Size of tiles for patch-wise alignment
} ICAParams;

// Function declarations
ImageGradients* init_ica(const Image* ref_img, const ICAParams* params);
void free_image_gradients(ImageGradients* grads);
HessianMatrix* compute_hessian(const ImageGradients* grads, int tile_size);
void free_hessian_matrix(HessianMatrix* hessian);

// Main ICA function
AlignmentMap* refine_alignment_ica(const Image* ref_img, const Image* alt_img,
                                 const ImageGradients* grads,
                                 const HessianMatrix* hessian,
                                 const AlignmentMap* initial_alignment,
                                 const ICAParams* params);

// Utility functions
void compute_image_gradients(const Image* img, ImageGradients* grads, float sigma_blur);
void solve_2x2_system(const float* A, const float* b, float* x);

#endif // ICA_H 