#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "utils.h"
#include "block_matching.h"

// Default parameters
#define DEFAULT_THREADS 16
#define EPSILON_DIV 1e-10f

// Math utilities
float clamp(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

// Compute Mean Squared Error between two images
float compute_mse(const Image* img1, const Image* img2) {
    if (!img1 || !img2 || 
        img1->height != img2->height || 
        img1->width != img2->width || 
        img1->channels != img2->channels) {
        return -1.0f;
    }

    double sum_sq = 0.0;
    size_t size = img1->height * img1->width * img1->channels;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sum_sq)
    #endif
    for (size_t i = 0; i < size; i++) {
        float diff = img1->data[i] - img2->data[i];
        sum_sq += diff * diff;
    }

    return (float)(sum_sq / size);
}

// Element-wise division of two images
void divide_images(Image* num, const Image* den) {
    if (!num || !den || 
        num->height != den->height || 
        num->width != den->width || 
        num->channels != den->channels) {
        return;
    }

    size_t size = num->height * num->width * num->channels;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < size; i++) {
        float denominator = den->data[i];
        if (fabsf(denominator) > EPSILON_DIV) {
            num->data[i] /= denominator;
        } else {
            num->data[i] = 0.0f;  // Handle division by zero
        }
    }
}

// Add two images (A += B)
void add_images(Image* A, const Image* B) {
    if (!A || !B || 
        A->height != B->height || 
        A->width != B->width || 
        A->channels != B->channels) {
        return;
    }

    size_t size = A->height * A->width * A->channels;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < size; i++) {
        A->data[i] += B->data[i];
    }
}

// Memory management utilities
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    #ifdef _WIN32
        ptr = _aligned_malloc(size, alignment);
    #else
        if (posix_memalign(&ptr, alignment, size) != 0) {
            ptr = NULL;
        }
    #endif
    return ptr;
}

void aligned_free(void* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// Save image to file
bool save_image(const char* filename, const Image* image) {
    if (!filename || !image) return false;

    // Convert float data to 8-bit
    unsigned char* data = (unsigned char*)malloc(image->height * image->width * image->channels);
    if (!data) return false;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3)
    #endif
    for (int y = 0; y < image->height; y++) {
        for (int x = 0; x < image->width; x++) {
            for (int c = 0; c < image->channels; c++) {
                int idx = (y * image->width + x) * image->channels + c;
                float val = clamp(image->data[idx], 0.0f, 1.0f);
                data[idx] = (unsigned char)(val * 255.0f + 0.5f);
            }
        }
    }

    bool success = stbi_write_png(filename, image->width, image->height, 
                                image->channels, data, 
                                image->width * image->channels) != 0;

    free(data);
    return success;
}

// Load image from file
Image* load_image(const char* filename) {
    if (!filename) return NULL;

    int width, height, channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) return NULL;

    Image* image = create_image_channels(height, width, channels);
    if (!image) {
        stbi_image_free(data);
        return NULL;
    }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3)
    #endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                image->data[idx] = data[idx] / 255.0f;
            }
        }
    }

    stbi_image_free(data);
    return image;
}

// Create grayscale image from color image
Image* create_grayscale(const Image* color_img) {
    if (!color_img || color_img->channels < 3) return NULL;

    Image* gray = create_image_channels(color_img->height, color_img->width, 1);
    if (!gray) return NULL;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int y = 0; y < color_img->height; y++) {
        for (int x = 0; x < color_img->width; x++) {
            int color_idx = (y * color_img->width + x) * color_img->channels;
            int gray_idx = y * color_img->width + x;
            // Convert to grayscale using standard weights
            gray->data[gray_idx] = 0.299f * color_img->data[color_idx] +
                                 0.587f * color_img->data[color_idx + 1] +
                                 0.114f * color_img->data[color_idx + 2];
        }
    }

    return gray;
}

// OpenMP helper to get optimal number of threads
int get_optimal_threads(void) {
    #ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    return (max_threads > DEFAULT_THREADS) ? DEFAULT_THREADS : max_threads;
    #else
    return 1;
    #endif
}

// Set number of threads for parallel processing
void set_num_threads(int num_threads) {
    #ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    #else
    (void)num_threads; // Avoid unused parameter warning
    #endif
}

// Get current thread ID
int get_thread_id(void) {
    #ifdef _OPENMP
    return omp_get_thread_num();
    #else
    return 0;
    #endif
}

// Check if running in parallel region
bool is_parallel(void) {
    #ifdef _OPENMP
    return omp_in_parallel() != 0;
    #else
    return false;
    #endif
}

// Timer functions
double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Print progress message
void print_progress(const char* message, int verbose_level) {
    if (verbose_level > 0) {
        printf("%s\n", message);
    }
}

// Parameter management
AlignmentParams* create_default_params(void) {
    AlignmentParams* params = (AlignmentParams*)malloc(sizeof(AlignmentParams));
    if (!params) return NULL;

    params->num_pyramid_levels = MAX_PYRAMID_LEVELS;
    
    // Allocate arrays
    params->factors = (int*)malloc(sizeof(int) * MAX_PYRAMID_LEVELS);
    params->tile_sizes = (int*)malloc(sizeof(int) * MAX_PYRAMID_LEVELS);
    params->search_radii = (int*)malloc(sizeof(int) * MAX_PYRAMID_LEVELS);
    params->use_l1_dist = (bool*)malloc(sizeof(bool) * MAX_PYRAMID_LEVELS);

    if (!params->factors || !params->tile_sizes || 
        !params->search_radii || !params->use_l1_dist) {
        free_alignment_params(params);
        return NULL;
    }

    // Set default values
    int default_factors[] = {1, 2, 4, 4};
    int default_search_radii[] = {1, 4, 4, 4};
    bool default_use_l1[] = {true, false, false, false};

    memcpy(params->factors, default_factors, sizeof(default_factors));
    memcpy(params->search_radii, default_search_radii, sizeof(default_search_radii));
    memcpy(params->use_l1_dist, default_use_l1, sizeof(default_use_l1));

    // Set tile sizes
    params->tile_size = DEFAULT_TILE_SIZE;
    for (int i = 0; i < MAX_PYRAMID_LEVELS; i++) {
        params->tile_sizes[i] = (i == MAX_PYRAMID_LEVELS - 1) ? 
                               params->tile_size / 2 : params->tile_size;
    }

    // ICA parameters
    params->sigma_blur = 0.0f;
    params->num_iterations = 3;

    return params;
}

void free_alignment_params(AlignmentParams* params) {
    if (params) {
        free(params->factors);
        free(params->tile_sizes);
        free(params->search_radii);
        free(params->use_l1_dist);
        free(params);
    }
}