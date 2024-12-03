#ifndef UTILS_IMAGE_H
#define UTILS_IMAGE_H

#include "utils.h"

// Image orientation enumeration (EXIF standard)
typedef enum {
    ORIENTATION_NORMAL = 1,
    ORIENTATION_MIRROR_HORIZONTAL = 2,
    ORIENTATION_ROTATE_180 = 3,
    ORIENTATION_MIRROR_VERTICAL = 4,
    ORIENTATION_MIRROR_HORIZONTAL_ROTATE_270 = 5,
    ORIENTATION_ROTATE_90 = 6,
    ORIENTATION_MIRROR_HORIZONTAL_ROTATE_90 = 7,
    ORIENTATION_ROTATE_270 = 8
} ImageOrientation;

// Denoising parameters
typedef struct {
    char* mode;            // Denoising mode ("grey" or "color")
    float scale;           // Scale factor
    float radius_max;      // Maximum radius for neighborhood
    int max_frame_count;   // Maximum number of frames to consider
    float sigma_max;       // Maximum sigma for Gaussian denoising
} DenoisingParams;

// Function declarations
void apply_orientation(Image* image, ImageOrientation orientation);
Image* compute_grey_image(const Image* image, const char* method);

// Denoising functions
Image* frame_count_denoising_gauss(const Image* image, const float* r_acc,
                                  const DenoisingParams* params);
Image* frame_count_denoising_median(const Image* image, const float* r_acc,
                                  const DenoisingParams* params);

// FFT operations
void fft_lowpass(float* image, int height, int width);
void fft2d(float* data, int height, int width, bool inverse);
void fftshift(float* data, int height, int width);

// Quality metrics
float compute_rmse(const Image* image1, const Image* image2);
float compute_psnr(const Image* clean, const Image* noisy);

// Helper functions
float denoise_power_gauss(float r_acc, float sigma_max, float r_max);
float denoise_power_median(float r_acc, float radius_max, float max_frame_count);
float denoise_power_merge(float r_acc, float power_max, float max_frame_count);
float denoise_range_merge(float r_acc, float rad_max, float max_frame_count);

#endif // UTILS_IMAGE_H 