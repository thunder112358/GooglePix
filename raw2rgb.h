#ifndef RAW2RGB_H
#define RAW2RGB_H

#include "utils.h"

// Color correction matrix structure
typedef struct {
    float matrix[3][3];    // 3x3 color transformation matrix
} ColorMatrix;

// Raw processing parameters
typedef struct {
    bool do_color_correction;  // Whether to apply color correction
    bool do_tonemapping;       // Whether to apply tone mapping
    bool do_gamma;             // Whether to apply gamma correction
    bool do_sharpening;        // Whether to apply sharpening
    bool do_devignette;        // Whether to apply devignetting
    float gamma;               // Gamma value for correction
    float sharpening_sigma;    // Sigma for sharpening kernel
    float sharpening_amount;   // Amount of sharpening to apply
} RawProcessingParams;

// Function declarations
ColorMatrix* get_xyz2cam_from_exif(const char* image_path);
ColorMatrix* get_random_ccm(void);
void free_color_matrix(ColorMatrix* matrix);

// Color space conversions
void apply_ccm(const Image* input, const ColorMatrix* ccm, Image* output);
void rgb_to_xyz(const Image* rgb, Image* xyz);
void xyz_to_rgb(const Image* xyz, Image* rgb);

// Tone mapping and corrections
void gamma_compression(Image* image, float gamma);
void gamma_expansion(Image* image, float gamma);
void apply_smoothstep(Image* image);
void invert_smoothstep(Image* image);

// Raw processing pipeline
Image* process_raw_image(const Image* raw_image,
                        const ColorMatrix* xyz2cam,
                        const RawProcessingParams* params);

// Helper functions
void apply_gains(Image* image, float red_gain, float blue_gain, float rgb_gain);
void safe_invert_gains(Image* image, float red_gain, float blue_gain, float rgb_gain);
void sharpen_image(Image* image, float sigma, float amount);
void apply_devignetting(Image* image, float* vignette_map);

#endif // RAW2RGB_H 