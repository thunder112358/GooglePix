#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include "utils_image.h"
#include "linalg.h"

// Apply orientation transformation to image
void apply_orientation(Image* image, ImageOrientation orientation) {
    if (!image || !image->data) return;

    const int height = image->height;
    const int width = image->width;
    const int channels = image->channels;
    const size_t size = height * width * channels;

    float* temp = (float*)malloc(size * sizeof(float));
    if (!temp) return;

    switch (orientation) {
        case ORIENTATION_NORMAL:
            // No change needed
            free(temp);
            return;

        case ORIENTATION_MIRROR_HORIZONTAL:
            #pragma omp parallel for collapse(3)
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        temp[(y * width + x) * channels + c] = 
                            image->data[(y * width + (width - 1 - x)) * channels + c];
                    }
                }
            }
            break;

        case ORIENTATION_ROTATE_180:
            #pragma omp parallel for collapse(3)
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        temp[(y * width + x) * channels + c] = 
                            image->data[((height - 1 - y) * width + (width - 1 - x)) * channels + c];
                    }
                }
            }
            break;

        case ORIENTATION_MIRROR_VERTICAL:
            #pragma omp parallel for collapse(3)
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        temp[(y * width + x) * channels + c] = 
                            image->data[((height - 1 - y) * width + x) * channels + c];
                    }
                }
            }
            break;

        case ORIENTATION_MIRROR_HORIZONTAL_ROTATE_270:
        case ORIENTATION_ROTATE_90:
        case ORIENTATION_MIRROR_HORIZONTAL_ROTATE_90:
        case ORIENTATION_ROTATE_270: {
            // For rotations, we need to swap dimensions
            Image* rotated = (Image*)malloc(sizeof(Image));
            if (!rotated) {
                free(temp);
                return;
            }
            rotated->width = height;
            rotated->height = width;
            rotated->channels = channels;
            rotated->data = (float*)malloc(size * sizeof(float));
            if (!rotated->data) {
                free(rotated);
                free(temp);
                return;
            }

            #pragma omp parallel for collapse(3)
            for (int c = 0; c < channels; c++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int new_x, new_y;
                        switch (orientation) {
                            case ORIENTATION_ROTATE_90:
                                new_x = height - 1 - y;
                                new_y = x;
                                break;
                            case ORIENTATION_ROTATE_270:
                                new_x = y;
                                new_y = width - 1 - x;
                                break;
                            case ORIENTATION_MIRROR_HORIZONTAL_ROTATE_90:
                                new_x = y;
                                new_y = x;
                                break;
                            case ORIENTATION_MIRROR_HORIZONTAL_ROTATE_270:
                                new_x = height - 1 - y;
                                new_y = width - 1 - x;
                                break;
                            default:
                                new_x = x;
                                new_y = y;
                        }
                        rotated->data[(new_y * rotated->width + new_x) * channels + c] = 
                            image->data[(y * width + x) * channels + c];
                    }
                }
            }

            // Update image dimensions and data
            free(image->data);
            image->width = rotated->width;
            image->height = rotated->height;
            image->data = rotated->data;
            free(rotated);
            free(temp);
            return;
        }
    }

    // Copy back the transformed data
    memcpy(image->data, temp, size * sizeof(float));
    free(temp);
}

// Convert RGB image to grayscale
Image* compute_grey_image(const Image* image, const char* method) {
    if (!image || !method) return NULL;

    Image* grey = (Image*)malloc(sizeof(Image));
    if (!grey) return NULL;

    grey->height = image->height;
    grey->width = image->width;
    grey->channels = 1;
    grey->data = (float*)malloc(grey->height * grey->width * sizeof(float));

    if (!grey->data) {
        free(grey);
        return NULL;
    }

    if (strcmp(method, "FFT") == 0) {
        // FFT-based grayscale conversion
        fft_lowpass(image->data, grey->height, grey->width);
        memcpy(grey->data, image->data, grey->height * grey->width * sizeof(float));
    } else {
        // Standard luminance conversion
        const float r_weight = 0.2989f;
        const float g_weight = 0.5870f;
        const float b_weight = 0.1140f;

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < grey->height; y++) {
            for (int x = 0; x < grey->width; x++) {
                int rgb_idx = (y * grey->width + x) * image->channels;
                int grey_idx = y * grey->width + x;
                grey->data[grey_idx] = 
                    r_weight * image->data[rgb_idx] +
                    g_weight * image->data[rgb_idx + 1] +
                    b_weight * image->data[rgb_idx + 2];
            }
        }
    }

    return grey;
}

// Gaussian denoising based on frame count
Image* frame_count_denoising_gauss(const Image* image, const float* r_acc,
                                 const DenoisingParams* params) {
    if (!image || !r_acc || !params) return NULL;

    Image* denoised = (Image*)malloc(sizeof(Image));
    if (!denoised) return NULL;

    denoised->height = image->height;
    denoised->width = image->width;
    denoised->channels = image->channels;
    denoised->data = (float*)malloc(image->height * image->width * image->channels * sizeof(float));

    if (!denoised->data) {
        free(denoised);
        return NULL;
    }

    const int height = image->height;
    const int width = image->width;
    const int channels = image->channels;

    #pragma omp parallel for collapse(3)
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float r = r_acc[y * width + x];
                float sigma = denoise_power_gauss(r, params->sigma_max, params->radius_max);
                
                if (sigma > 1e-6f) {
                    // Apply local Gaussian blur
                    float sum = 0.0f;
                    float weight_sum = 0.0f;
                    int radius = (int)ceilf(3.0f * sigma);

                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                float dist = sqrtf(dx*dx + dy*dy);
                                float weight = expf(-0.5f * (dist*dist)/(sigma*sigma));
                                sum += weight * image->data[(ny * width + nx) * channels + c];
                                weight_sum += weight;
                            }
                        }
                    }
                    denoised->data[(y * width + x) * channels + c] = sum / weight_sum;
                } else {
                    denoised->data[(y * width + x) * channels + c] = 
                        image->data[(y * width + x) * channels + c];
                }
            }
        }
    }

    return denoised;
}

// Median denoising based on frame count
Image* frame_count_denoising_median(const Image* image, const float* r_acc,
                                  const DenoisingParams* params) {
    if (!image || !r_acc || !params) return NULL;

    Image* denoised = (Image*)malloc(sizeof(Image));
    if (!denoised) return NULL;

    denoised->height = image->height;
    denoised->width = image->width;
    denoised->channels = image->channels;
    denoised->data = (float*)malloc(image->height * image->width * image->channels * sizeof(float));

    if (!denoised->data) {
        free(denoised);
        return NULL;
    }

    const int height = image->height;
    const int width = image->width;
    const int channels = image->channels;
    const int max_window = 29 * 29;  // Maximum window size for median filter
    float* values = (float*)malloc(max_window * sizeof(float));
    
    if (!values) {
        free(denoised->data);
        free(denoised);
        return NULL;
    }

    #pragma omp parallel for collapse(3) private(values)
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float r = r_acc[y * width + x];
                int radius = (int)denoise_power_median(r, params->radius_max, params->max_frame_count);
                
                if (radius > 0) {
                    // Apply median filter
                    int count = 0;
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                values[count++] = image->data[(ny * width + nx) * channels + c];
                            }
                        }
                    }

                    // Sort values and take median
                    for (int i = 0; i < count-1; i++) {
                        for (int j = 0; j < count-i-1; j++) {
                            if (values[j] > values[j+1]) {
                                float temp = values[j];
                                values[j] = values[j+1];
                                values[j+1] = temp;
                            }
                        }
                    }
                    denoised->data[(y * width + x) * channels + c] = values[count/2];
                } else {
                    denoised->data[(y * width + x) * channels + c] = 
                        image->data[(y * width + x) * channels + c];
                }
            }
        }
    }

    free(values);
    return denoised;
}

// Helper functions for denoising power calculation
float denoise_power_gauss(float r_acc, float sigma_max, float r_max) {
    float r = fminf(r_acc, r_max);
    return sigma_max * (r_max - r) / r_max;
}

float denoise_power_median(float r_acc, float radius_max, float max_frame_count) {
    float r = fminf(r_acc, max_frame_count);
    return roundf(radius_max * (max_frame_count - r) / max_frame_count);
}

float denoise_power_merge(float r_acc, float power_max, float max_frame_count) {
    return (r_acc <= max_frame_count) ? power_max : 1.0f;
}

float denoise_range_merge(float r_acc, float rad_max, float max_frame_count) {
    const float rad_min = 1.0f;  // 3x3 window
    return (r_acc <= max_frame_count) ? rad_max : rad_min;
}

// Quality metrics
float compute_rmse(const Image* image1, const Image* image2) {
    if (!image1 || !image2 || 
        image1->height != image2->height || 
        image1->width != image2->width || 
        image1->channels != image2->channels) {
        return -1.0f;
    }

    const size_t size = image1->height * image1->width * image1->channels;
    double sum_sq = 0.0;

    #pragma omp parallel for reduction(+:sum_sq)
    for (size_t i = 0; i < size; i++) {
        float diff = image1->data[i] - image2->data[i];
        sum_sq += diff * diff;
    }

    return sqrtf(sum_sq / size);
}

float compute_psnr(const Image* clean, const Image* noisy) {
    if (!clean || !noisy || 
        clean->height != noisy->height || 
        clean->width != noisy->width || 
        clean->channels != noisy->channels) {
        return -1.0f;
    }

    float rmse = compute_rmse(clean, noisy);
    if (rmse <= 0.0f) return -1.0f;

    // Assuming images are in [0,1] range
    return 20.0f * log10f(1.0f / rmse);
} 