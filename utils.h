#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include "block_matching.h"
#include "common.h"
// Constants
#define DEFAULT_TILE_SIZE 16
#define MAX_PYRAMID_LEVELS 4

// Parameter structures
typedef struct {
    int verbose;  // Verbosity level (0-3)
} Options;

typedef struct {
    // Block matching parameters
    int num_pyramid_levels;
    int* factors;           // Downsampling factors for each level
    int* tile_sizes;        // Tile sizes for each level
    int* search_radii;      // Search radii for each level
    bool* use_l1_dist;      // true for L1 distance, false for L2

    // ICA parameters
    float sigma_blur;       // Gaussian blur sigma
    int num_iterations;     // Number of ICA iterations
    int tile_size;         // Base tile size
} AlignmentParams;

/*
// Forward declarations of structures
typedef struct Image Image;
typedef struct ImagePyramid ImagePyramid;

// Structure definitions
struct Image {
    float* data;
    int height;
    int width;
    int channels;
};

struct ImagePyramid {
    Image** images;
    int num_levels;
};
*/
// Function declarations
Image* create_image_channels(int height, int width, int channels);
Image* create_image(int height, int width);

// Image I/O functions
Image* load_image(const char* filename);
bool save_image(const char* filename, const Image* img);
Image* create_grayscale(const Image* color_img);

// Parameter handling
AlignmentParams* create_default_params(void);
void free_alignment_params(AlignmentParams* params);

// Utility functions
void print_progress(const char* message, int verbose_level);
double get_time(void);
void clip_value(float* value, float min, float max);

#endif // UTILS_H 