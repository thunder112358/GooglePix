#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "block_matching.h"

// Helper function declarations
static void compute_gaussian_kernel(float* kernel, int size, float sigma);
static void apply_gaussian_blur(Image* img, float sigma);

// Memory management functions
Image* create_image_channels(int height, int width, int channels) {
    Image* img = (Image*)malloc(sizeof(Image));
    if (!img) return NULL;

    img->height = height;
    img->width = width;
    img->channels = channels;
    img->data = (float*)calloc(height * width * channels, sizeof(float));

    if (!img->data) {
        free(img);
        return NULL;
    }

    return img;
}

Image* create_image(int height, int width) {
    return create_image_channels(height, width, 3);  // Default to 3 channels (RGB)
}

void free_image(Image* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

AlignmentMap* create_alignment_map(int height, int width, int patch_size) {
    AlignmentMap* map = (AlignmentMap*)malloc(sizeof(AlignmentMap));
    if (!map) return NULL;

    map->height = height;
    map->width = width;
    map->patch_size = patch_size;
    map->data = (Alignment*)calloc(height * width, sizeof(Alignment));
    
    if (!map->data) {
        free(map);
        return NULL;
    }

    return map;
}

void free_alignment_map(AlignmentMap* map) {
    if (map) {
        free(map->data);
        free(map);
    }
}

ImagePyramid* create_image_pyramid(int num_levels) {
    ImagePyramid* pyramid = (ImagePyramid*)malloc(sizeof(ImagePyramid));
    if (!pyramid) return NULL;

    pyramid->num_levels = num_levels;
    pyramid->levels = (Image**)calloc(num_levels, sizeof(Image*));
    pyramid->factors = (int*)malloc(sizeof(int) * num_levels);
    
    if (!pyramid->levels || !pyramid->factors) {
        free(pyramid->levels);
        free(pyramid->factors);
        free(pyramid);
        return NULL;
    }

    return pyramid;
}

void free_image_pyramid(ImagePyramid* pyramid) {
    if (pyramid) {
        for (int i = 0; i < pyramid->num_levels; i++) {
            free_image(pyramid->levels[i]);
        }
        free(pyramid->levels);
        free(pyramid->factors);
        free(pyramid);
    }
}

// Image processing functions
Image* pad_image(const Image* img, int pad_top, int pad_bottom, int pad_left, int pad_right) {
    int new_height = img->height + pad_top + pad_bottom;
    int new_width = img->width + pad_left + pad_right;
    
    Image* padded = create_image(new_height, new_width);
    if (!padded) return NULL;

    // Copy original image
    for (int y = 0; y < img->height; y++) {
        memcpy(&padded->data[(y + pad_top) * new_width + pad_left],
               &img->data[y * img->width],
               img->width * sizeof(pixel_t));
    }

    // Pad top and bottom
    for (int y = 0; y < pad_top; y++) {
        memcpy(&padded->data[y * new_width + pad_left],
               &img->data[0],
               img->width * sizeof(pixel_t));
    }
    for (int y = 0; y < pad_bottom; y++) {
        memcpy(&padded->data[(new_height - pad_bottom + y) * new_width + pad_left],
               &img->data[(img->height - 1) * img->width],
               img->width * sizeof(pixel_t));
    }

    // Pad left and right
    for (int y = 0; y < new_height; y++) {
        // Left padding
        for (int x = 0; x < pad_left; x++) {
            padded->data[y * new_width + x] = padded->data[y * new_width + pad_left];
        }
        // Right padding
        for (int x = 0; x < pad_right; x++) {
            padded->data[y * new_width + new_width - pad_right + x] = 
                padded->data[y * new_width + new_width - pad_right - 1];
        }
    }

    return padded;
}

Image* downsample_image(const Image* img, int factor) {
    if (factor <= 1) {
        Image* copy = create_image(img->height, img->width);
        if (copy) {
            memcpy(copy->data, img->data, sizeof(pixel_t) * img->height * img->width);
        }
        return copy;
    }

    int new_height = img->height / factor;
    int new_width = img->width / factor;
    Image* downsampled = create_image(new_height, new_width);
    if (!downsampled) return NULL;

    // Box filter downsampling
    float scale = 1.0f / (factor * factor);
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < factor; ky++) {
                for (int kx = 0; kx < factor; kx++) {
                    int orig_y = y * factor + ky;
                    int orig_x = x * factor + kx;
                    sum += img->data[orig_y * img->width + orig_x];
                }
            }
            downsampled->data[y * new_width + x] = sum * scale;
        }
    }

    return downsampled;
}

// Distance metric functions
float compute_patch_distance_l1(const Image* ref_img, const Image* target_img,
                              int ref_x, int ref_y, int target_x, int target_y,
                              int patch_size) {
    float dist = 0.0f;
    
    for (int y = 0; y < patch_size; y++) {
        for (int x = 0; x < patch_size; x++) {
            int rx = ref_x + x;
            int ry = ref_y + y;
            int tx = target_x + x;
            int ty = target_y + y;
            
            if (rx >= ref_img->width || ry >= ref_img->height ||
                tx >= target_img->width || ty >= target_img->height) {
                return FLT_MAX;
            }
            
            float diff = ref_img->data[ry * ref_img->width + rx] - 
                        target_img->data[ty * target_img->width + tx];
            dist += fabsf(diff);
        }
    }
    
    return dist;
}

float compute_patch_distance_l2(const Image* ref_img, const Image* target_img,
                              int ref_x, int ref_y, int target_x, int target_y,
                              int patch_size) {
    float dist = 0.0f;
    
    for (int y = 0; y < patch_size; y++) {
        for (int x = 0; x < patch_size; x++) {
            int rx = ref_x + x;
            int ry = ref_y + y;
            int tx = target_x + x;
            int ty = target_y + y;
            
            if (rx >= ref_img->width || ry >= ref_img->height ||
                tx >= target_img->width || ty >= target_img->height) {
                return FLT_MAX;
            }
            
            float diff = ref_img->data[ry * ref_img->width + rx] - 
                        target_img->data[ty * target_img->width + tx];
            dist += diff * diff;
        }
    }
    
    return dist;
}

// Core block matching functions
ImagePyramid* init_block_matching(const Image* ref_img, const BlockMatchingParams* params) {
    ImagePyramid* pyramid = create_image_pyramid(params->num_levels);
    if (!pyramid) return NULL;

    // Copy factors
    memcpy(pyramid->factors, params->factors, sizeof(int) * params->num_levels);

    // Create first level (possibly with padding)
    Image* padded = pad_image(ref_img, params->pad_top, params->pad_bottom,
                            params->pad_left, params->pad_right);
    if (!padded) {
        free_image_pyramid(pyramid);
        return NULL;
    }

    pyramid->levels[0] = padded;

    // Create subsequent levels
    for (int i = 1; i < params->num_levels; i++) {
        pyramid->levels[i] = downsample_image(pyramid->levels[i-1], params->factors[i]);
        if (!pyramid->levels[i]) {
            free_image_pyramid(pyramid);
            return NULL;
        }
    }

    return pyramid;
}

static void search_best_match(const Image* ref_img, const Image* target_img,
                            int patch_y, int patch_x, int patch_size,
                            int search_radius, bool use_l1,
                            float current_x, float current_y,
                            float* best_x, float* best_y) {
    float min_dist = FLT_MAX;
    int ref_patch_x = patch_x * patch_size;
    int ref_patch_y = patch_y * patch_size;

    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            int target_x = ref_patch_x + (int)current_x + dx;
            int target_y = ref_patch_y + (int)current_y + dy;
            
            float dist;
            if (use_l1) {
                dist = compute_patch_distance_l1(ref_img, target_img,
                                              ref_patch_x, ref_patch_y,
                                              target_x, target_y,
                                              patch_size);
            } else {
                dist = compute_patch_distance_l2(ref_img, target_img,
                                              ref_patch_x, ref_patch_y,
                                              target_x, target_y,
                                              patch_size);
            }

            if (dist < min_dist) {
                min_dist = dist;
                *best_x = current_x + dx;
                *best_y = current_y + dy;
            }
        }
    }
}

AlignmentMap* align_image_block_matching(const Image* img, const ImagePyramid* reference_pyramid,
                                       const BlockMatchingParams* params) {
    // Create pyramid for target image
    ImagePyramid* target_pyramid = init_block_matching(img, params);
    if (!target_pyramid) return NULL;

    AlignmentMap* current_alignment = NULL;
    
    // Process from coarsest to finest level
    for (int level = params->num_levels - 1; level >= 0; level--) {
        int tile_size = params->tile_sizes[level];
        int n_patches_y = reference_pyramid->levels[level]->height / tile_size;
        int n_patches_x = reference_pyramid->levels[level]->width / tile_size;

        // Create or upscale alignment map
        AlignmentMap* level_alignment;
        if (current_alignment) {
            level_alignment = create_alignment_map(n_patches_y, n_patches_x, tile_size);
            if (!level_alignment) {
                free_image_pyramid(target_pyramid);
                free_alignment_map(current_alignment);
                return NULL;
            }
            
            // Upscale previous alignment
            float scale = (float)params->factors[level];
            for (int y = 0; y < level_alignment->height; y++) {
                for (int x = 0; x < level_alignment->width; x++) {
                    int prev_y = y / scale;
                    int prev_x = x / scale;
                    level_alignment->data[y * level_alignment->width + x].x = 
                        current_alignment->data[prev_y * current_alignment->width + prev_x].x * scale;
                    level_alignment->data[y * level_alignment->width + x].y = 
                        current_alignment->data[prev_y * current_alignment->width + prev_x].y * scale;
                }
            }
            free_alignment_map(current_alignment);
        } else {
            level_alignment = create_alignment_map(n_patches_y, n_patches_x, tile_size);
            if (!level_alignment) {
                free_image_pyramid(target_pyramid);
                return NULL;
            }
        }

        // Perform block matching at current level
        for (int py = 0; py < n_patches_y; py++) {
            for (int px = 0; px < n_patches_x; px++) {
                float current_x = level_alignment->data[py * n_patches_x + px].x;
                float current_y = level_alignment->data[py * n_patches_x + px].y;
                
                float best_x, best_y;
                search_best_match(reference_pyramid->levels[level],
                                target_pyramid->levels[level],
                                py, px, tile_size,
                                params->search_radii[level],
                                params->use_l1_dist[level],
                                current_x, current_y,
                                &best_x, &best_y);
                
                level_alignment->data[py * n_patches_x + px].x = best_x;
                level_alignment->data[py * n_patches_x + px].y = best_y;
            }
        }

        current_alignment = level_alignment;
    }

    free_image_pyramid(target_pyramid);
    return current_alignment;
} 