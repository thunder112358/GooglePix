#ifndef UTILS_DNG_H
#define UTILS_DNG_H

#include <stdbool.h>
#include "utils.h"
#include "raw2rgb.h"

// DNG metadata structure
typedef struct {
    int width;
    int height;
    int bits_per_sample;
    int samples_per_pixel;
    int photometric;
    int compression;
    int rows_per_strip;
    int planar_config;
    ColorMatrix* color_matrix1;
    ColorMatrix* color_matrix2;
    float* as_shot_neutral;
    float baseline_exposure;
    int calibration_illuminant1;
    int calibration_illuminant2;
} DNGMetadata;

// Supported photometric interpretations
#define PHOTO_WHITE_IS_ZERO 0
#define PHOTO_BLACK_IS_ZERO 1
#define PHOTO_RGB 2
#define PHOTO_RGB_PALETTE 3
#define PHOTO_TRANSPARENCY 4
#define PHOTO_CMYK 5
#define PHOTO_YCBCR 6
#define PHOTO_CIELAB 8
#define PHOTO_ICCLAB 9
#define PHOTO_ITULAB 10
#define PHOTO_CFA 32803
#define PHOTO_PIXAR_LOGL 32844
#define PHOTO_PIXAR_LOGLUV 32845
#define PHOTO_SEQUENTIAL_COLOR 32892
#define PHOTO_LINEAR_RAW 34892
#define PHOTO_DEPTH_MAP 51177
#define PHOTO_SEMANTIC_MASK 52527

// Function declarations
Image* load_dng(const char* filename, DNGMetadata* metadata);
bool save_as_dng(const Image* image, const char* ref_dng_path, const char* output_path);

// Metadata handling
DNGMetadata* read_dng_metadata(const char* filename);
void free_dng_metadata(DNGMetadata* metadata);

// DNG validation and conversion
bool validate_dng(const char* filename);
bool convert_to_dng(const char* input_path, const char* output_path, const DNGMetadata* metadata);

// Helper functions
bool is_supported_photometric(int photometric);
void copy_dng_tags(const char* src_path, const char* dst_path);
bool run_dng_validate(const char* filename);

#endif // UTILS_DNG_H 