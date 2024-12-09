/**
 * @file ica.h
 * @brief Iterative Closest Algorithm (ICA) implementation for image alignment refinement
 */

#ifndef ICA_H
#define ICA_H

#include "block_matching.h"
#include "common.h"

// Gradient structure matching Python implementation
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

// Extended parameters structure to match Python
typedef struct {
    float sigma_blur;     // Gaussian blur sigma (0 means no blur)
    int num_iterations;   // Number of Kanade iterations
    int tile_size;       // Size of tiles for patch-wise alignment
    bool debug_mode;     // Enable debug output
    bool save_iterations; // Save intermediate results
    char* debug_dir;     // Directory for debug output
    float convergence_threshold; // Stop iterations if change is below this
} ICAParams;

// Debug information structure
typedef struct {
    int iteration;
    float* flow_x;
    float* flow_y;
    float* residuals;
    int height;
    int width;
} ICADebugInfo;

// Add these declarations to ica.h
typedef struct {
    float* data;
    int height;
    int width;
    float min_error;
    float max_error;
    float mean_error;
} ErrorMap;

// Region analysis structure
typedef struct {
    float mean_error;
    float max_error;
    float min_error;
    int x;          // Region top-left x
    int y;          // Region top-left y
    int width;
    int height;
} RegionStats;

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

// Debug functions
void save_ica_debug_info(const ICADebugInfo* debug_info, const char* filename);
void free_ica_debug_info(ICADebugInfo* debug_info);

// Error visualization functions
ErrorMap* create_error_map(const Image* ref_img, const Image* target_img, 
                          const AlignmentMap* alignment);
void save_error_map_visualization(const ErrorMap* error_map, const char* filename);
void free_error_map(ErrorMap* error_map);

// Add these declarations
void analyze_error_maps(const ErrorMap* bm_error, const ErrorMap* ica_error);

// Region analysis functions
void analyze_regions(const ErrorMap* error_map, int num_regions_x, int num_regions_y);
RegionStats compute_region_stats(const ErrorMap* error_map, int x, int y, int width, int height);

// Add to ica.h
// Function to create warped image based on alignment
Image* create_warped_image(const Image* src_img, const AlignmentMap* alignment);

#endif // ICA_H 