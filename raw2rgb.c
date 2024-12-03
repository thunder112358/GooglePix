#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "raw2rgb.h"
#include "utils.h"

// Constants for Bayer pattern handling
#define BAYER_PATTERN_RGGB 0
#define BAYER_PATTERN_BGGR 1
#define BAYER_PATTERN_GRBG 2
#define BAYER_PATTERN_GBRG 3

// Helper function to get Bayer pattern type from string
static int get_bayer_pattern_type(const char* pattern) {
    if (!pattern) return BAYER_PATTERN_RGGB;  // Default
    
    if (strcmp(pattern, "RGGB") == 0) return BAYER_PATTERN_RGGB;
    if (strcmp(pattern, "BGGR") == 0) return BAYER_PATTERN_BGGR;
    if (strcmp(pattern, "GRBG") == 0) return BAYER_PATTERN_GRBG;
    if (strcmp(pattern, "GBRG") == 0) return BAYER_PATTERN_GBRG;
    
    return BAYER_PATTERN_RGGB;  // Default if unknown
}

// Helper function to get pixel value with bounds checking
static float get_pixel_safe(const float* data, int x, int y, int width, int height) {
    if (x < 0) x = 0;
    if (x >= width) x = width - 1;
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;
    
    return data[y * width + x];
}

// Demosaic RAW image using bilinear interpolation
Image* demosaic_raw(const float* raw_data, int width, int height, const char* bayer_pattern) {
    Image* rgb = create_image_channels(height, width, 3);
    if (!rgb) return NULL;
    
    int pattern_type = get_bayer_pattern_type(bayer_pattern);
    
    // Determine pattern offsets
    int r_off_y = 0, r_off_x = 0;
    int b_off_y = 0, b_off_x = 0;
    
    switch (pattern_type) {
        case BAYER_PATTERN_RGGB:
            r_off_y = 0; r_off_x = 0;
            b_off_y = 1; b_off_x = 1;
            break;
        case BAYER_PATTERN_BGGR:
            r_off_y = 1; r_off_x = 1;
            b_off_y = 0; b_off_x = 0;
            break;
        case BAYER_PATTERN_GRBG:
            r_off_y = 0; r_off_x = 1;
            b_off_y = 1; b_off_x = 0;
            break;
        case BAYER_PATTERN_GBRG:
            r_off_y = 1; r_off_x = 0;
            b_off_y = 0; b_off_x = 1;
            break;
    }

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = 0.0f, g = 0.0f, b = 0.0f;
            int idx = (y * width + x) * 3;
            
            // Red channel
            if ((y % 2) == r_off_y && (x % 2) == r_off_x) {
                // Red pixel
                r = raw_data[y * width + x];
                // Green at red pixel
                g = (get_pixel_safe(raw_data, x, y-1, width, height) +
                     get_pixel_safe(raw_data, x-1, y, width, height) +
                     get_pixel_safe(raw_data, x+1, y, width, height) +
                     get_pixel_safe(raw_data, x, y+1, width, height)) * 0.25f;
                // Blue at red pixel
                b = (get_pixel_safe(raw_data, x-1, y-1, width, height) +
                     get_pixel_safe(raw_data, x+1, y-1, width, height) +
                     get_pixel_safe(raw_data, x-1, y+1, width, height) +
                     get_pixel_safe(raw_data, x+1, y+1, width, height)) * 0.25f;
            }
            // Blue channel
            else if ((y % 2) == b_off_y && (x % 2) == b_off_x) {
                // Red at blue pixel
                r = (get_pixel_safe(raw_data, x-1, y-1, width, height) +
                     get_pixel_safe(raw_data, x+1, y-1, width, height) +
                     get_pixel_safe(raw_data, x-1, y+1, width, height) +
                     get_pixel_safe(raw_data, x+1, y+1, width, height)) * 0.25f;
                // Green at blue pixel
                g = (get_pixel_safe(raw_data, x, y-1, width, height) +
                     get_pixel_safe(raw_data, x-1, y, width, height) +
                     get_pixel_safe(raw_data, x+1, y, width, height) +
                     get_pixel_safe(raw_data, x, y+1, width, height)) * 0.25f;
                // Blue pixel
                b = raw_data[y * width + x];
            }
            // Green channel (in red row)
            else if ((y % 2) == r_off_y) {
                // Red at green pixel
                r = (get_pixel_safe(raw_data, x-1, y, width, height) +
                     get_pixel_safe(raw_data, x+1, y, width, height)) * 0.5f;
                // Green pixel
                g = raw_data[y * width + x];
                // Blue at green pixel
                b = (get_pixel_safe(raw_data, x, y-1, width, height) +
                     get_pixel_safe(raw_data, x, y+1, width, height)) * 0.5f;
            }
            // Green channel (in blue row)
            else {
                // Red at green pixel
                r = (get_pixel_safe(raw_data, x, y-1, width, height) +
                     get_pixel_safe(raw_data, x, y+1, width, height)) * 0.5f;
                // Green pixel
                g = raw_data[y * width + x];
                // Blue at green pixel
                b = (get_pixel_safe(raw_data, x-1, y, width, height) +
                     get_pixel_safe(raw_data, x+1, y, width, height)) * 0.5f;
            }
            
            rgb->data[idx] = r;
            rgb->data[idx + 1] = g;
            rgb->data[idx + 2] = b;
        }
    }
    
    return rgb;
}

// Apply white balance to RGB image
void apply_white_balance(Image* rgb, const float wb_gains[3]) {
    if (!rgb || !wb_gains) return;
    
    size_t size = rgb->width * rgb->height;
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        rgb->data[i * 3] *= wb_gains[0];     // R
        rgb->data[i * 3 + 1] *= wb_gains[1]; // G
        rgb->data[i * 3 + 2] *= wb_gains[2]; // B
    }
}

// Convert RAW data to RGB image
Image* raw_to_rgb(const float* raw_data, int width, int height, 
                 const char* bayer_pattern, const float wb_gains[3]) {
    // First demosaic the RAW data
    Image* rgb = demosaic_raw(raw_data, width, height, bayer_pattern);
    if (!rgb) return NULL;
    
    // Apply white balance if gains are provided
    if (wb_gains) {
        apply_white_balance(rgb, wb_gains);
    }
    
    return rgb;
}

// Convert RGB image to RAW data based on Bayer pattern
float* rgb_to_raw(const Image* rgb, const char* bayer_pattern) {
    if (!rgb || rgb->channels != 3) return NULL;
    
    float* raw_data = (float*)malloc(rgb->width * rgb->height * sizeof(float));
    if (!raw_data) return NULL;
    
    int pattern_type = get_bayer_pattern_type(bayer_pattern);
    int r_off_y = 0, r_off_x = 0;
    int b_off_y = 0, b_off_x = 0;
    
    // Set pattern offsets
    switch (pattern_type) {
        case BAYER_PATTERN_RGGB:
            r_off_y = 0; r_off_x = 0;
            b_off_y = 1; b_off_x = 1;
            break;
        case BAYER_PATTERN_BGGR:
            r_off_y = 1; r_off_x = 1;
            b_off_y = 0; b_off_x = 0;
            break;
        case BAYER_PATTERN_GRBG:
            r_off_y = 0; r_off_x = 1;
            b_off_y = 1; b_off_x = 0;
            break;
        case BAYER_PATTERN_GBRG:
            r_off_y = 1; r_off_x = 0;
            b_off_y = 0; b_off_x = 1;
            break;
    }
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rgb->height; y++) {
        for (int x = 0; x < rgb->width; x++) {
            int rgb_idx = (y * rgb->width + x) * 3;
            int raw_idx = y * rgb->width + x;
            
            if ((y % 2) == r_off_y && (x % 2) == r_off_x) {
                raw_data[raw_idx] = rgb->data[rgb_idx];     // R
            }
            else if ((y % 2) == b_off_y && (x % 2) == b_off_x) {
                raw_data[raw_idx] = rgb->data[rgb_idx + 2]; // B
            }
            else {
                raw_data[raw_idx] = rgb->data[rgb_idx + 1]; // G
            }
        }
    }
    
    return raw_data;
}

// Helper function to check if coordinates are within bounds
bool is_valid_coord(int x, int y, int width, int height) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

// Apply black level correction to RAW data
void apply_black_level(float* raw_data, int width, int height, float black_level) {
    if (!raw_data) return;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            raw_data[idx] = fmaxf(raw_data[idx] - black_level, 0.0f);
        }
    }
}

// Normalize RAW data to [0,1] range
void normalize_raw(float* raw_data, int width, int height, float white_level) {
    if (!raw_data || white_level <= 0.0f) return;
    
    float scale = 1.0f / white_level;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            raw_data[idx] *= scale;
            raw_data[idx] = fminf(fmaxf(raw_data[idx], 0.0f), 1.0f);
        }
    }
} 