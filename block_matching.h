/**
 * @file block_matching.h
 * @brief CPU implementation of block matching for image alignment
 */

#ifndef BLOCK_MATCHING_H
#define BLOCK_MATCHING_H

#include <stdint.h>
#include <stdbool.h>
#include "common.h"

// Block matching specific structures
typedef struct {
    float x;  // Horizontal alignment
    float y;  // Vertical alignment
} Alignment;

typedef struct {
    Alignment* data;
    int height;      // Number of patches in y direction
    int width;       // Number of patches in x direction
    int patch_size;  // Size of each patch
} AlignmentMap;

// Image pyramid for multi-scale processing
typedef struct {
    Image** levels;
    int num_levels;
    int* factors;    // Keep as int* since we use integer factors
} ImagePyramid;

// Filter types for downsampling
typedef enum {
    FILTER_BOX,
    FILTER_GAUSSIAN
} FilterMode;

// Block matching parameters
typedef struct {
    // Pyramid parameters
    int num_levels;
    int* factors;           // Downsampling factors for each level
    int* tile_sizes;        // Tile sizes for each level
    int* search_radii;      // Search radii for each level
    bool* use_l1_dist;      // true for L1 distance, false for L2

    // Padding parameters
    int pad_top;
    int pad_bottom;
    int pad_left;
    int pad_right;

    // Other parameters
    bool debug_mode;
    int max_iterations;
    FilterMode filter_mode;     // Type of filter to use for downsampling
    float gaussian_sigma;       // Sigma for Gaussian filter when used
    bool save_refinement_debug;  // Whether to save refinement visualization
    char* debug_output_dir;      // Directory for debug output
} BlockMatchingParams;

// Function declarations

// Core block matching functions
ImagePyramid* init_block_matching(const Image* ref_img, const BlockMatchingParams* params);
AlignmentMap* align_image_block_matching(const Image* img, const ImagePyramid* reference_pyramid, 
                                       const BlockMatchingParams* params);

// Pyramid operations
Image* downsample_image(const Image* img, int factor, FilterMode filter_mode, float gaussian_sigma);
Image* pad_image(const Image* img, int pad_top, int pad_bottom, int pad_left, int pad_right);

// Distance metrics
float compute_patch_distance_l1(const Image* ref_img, const Image* target_img,
                              int ref_x, int ref_y, int target_x, int target_y,
                              int patch_size);
float compute_patch_distance_l2(const Image* ref_img, const Image* target_img,
                              int ref_x, int ref_y, int target_x, int target_y,
                              int patch_size);

// Memory management
Image* create_image(int height, int width);
void free_image(Image* img);
AlignmentMap* create_alignment_map(int height, int width, int patch_size);
void free_alignment_map(AlignmentMap* alignments);
ImagePyramid* create_image_pyramid(int num_levels);
void free_image_pyramid(ImagePyramid* pyramid);

// Utility functions
void copy_image_region(const Image* src, Image* dst, 
                      int src_x, int src_y, int dst_x, int dst_y,
                      int width, int height);
void upsample_alignment_map(const AlignmentMap* src, AlignmentMap* dst, 
                          int scale_factor);

// Add this declaration if it's not already there
Image* create_image_channels(int height, int width, int channels);

#define MAX_PYRAMID_LEVELS 10  // or whatever maximum makes sense for your use case

// Add after other struct definitions
typedef struct {
    int patch_x;
    int patch_y;
    float original_x;
    float original_y;
    float vertical_x;
    float vertical_y;
    float horizontal_x;
    float horizontal_y;
    float final_x;
    float final_y;
    float distances[3];  // distances for each candidate
} RefinementDebugInfo;

#endif // BLOCK_MATCHING_H 
