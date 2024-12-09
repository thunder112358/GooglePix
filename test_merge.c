#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "merge.h"

// Helper function to create test image
static Image* create_test_image(int height, int width, int channels) {
    Image* img = (Image*)malloc(sizeof(Image));
    if (!img) return NULL;
    
    img->height = height;
    img->width = width;
    img->channels = channels;
    img->data = (float*)malloc(height * width * channels * sizeof(float));
    
    if (!img->data) {
        free(img);
        return NULL;
    }
    
    // Create test pattern (sine wave)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = x - width/2;
            float dy = y - height/2;
            float r = sqrtf(dx*dx + dy*dy);
            float val = (sinf(r * 0.2f) + 1.0f) * 0.5f;  // Range [0,1]
            
            for (int c = 0; c < channels; c++) {
                img->data[(y * width + x) * channels + c] = val;
            }
        }
    }
    
    return img;
}

// Helper function to free image
static void free_test_image(Image* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// Helper function to save image as PPM
static void save_image_ppm(const Image* img, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    
    unsigned char* rgb = (unsigned char*)malloc(img->width * img->height * 3);
    if (rgb) {
        for (int i = 0; i < img->width * img->height; i++) {
            for (int c = 0; c < 3; c++) {
                int idx = i * img->channels + (c % img->channels);
                rgb[i * 3 + c] = (unsigned char)(img->data[idx] * 255.0f);
            }
        }
        fwrite(rgb, 1, img->width * img->height * 3, fp);
        free(rgb);
    }
    
    fclose(fp);
}

int main() {
    // Test parameters
    int width = 64;
    int height = 64;
    int channels = 1;  // Start with grayscale
    float scale = 2.0f;  // 2x upscaling
    
    // Create test images
    Image* ref_img = create_test_image(height, width, channels);
    Image* target_img = create_test_image(height, width, channels);
    if (!ref_img || !target_img) {
        printf("Failed to create test images\n");
        return 1;
    }
    
    // Create merge parameters
    MergeParams params = {
        .power_max = 1.0f,
        .max_frame_count = 8,
        .radius_max = 2.0f,
        .noise_sigma = 0.1f,
        .use_robustness = false,
        .use_kernels = false,
        .scale = scale,
        .bayer_mode = false,
        .iso_kernel = true,
        .tile_size = 8,
        .cfa_pattern = NULL,
        .use_acc_rob = false,
        .max_multiplier = 2.0f
    };
    
    // Create accumulator for upscaled output
    int out_height = (int)(height * scale);
    int out_width = (int)(width * scale);
    MergeAccumulator* acc = create_merge_accumulator(out_height, out_width, channels);
    if (!acc) {
        printf("Failed to create accumulator\n");
        free_test_image(ref_img);
        free_test_image(target_img);
        return 1;
    }
    
    // Test merge reference
    printf("Merging reference image...\n");
    merge_reference(ref_img, acc, &params);
    
    // Test merge target
    printf("Merging target image...\n");
    merge_image(target_img, NULL, NULL, acc, &params);
    
    // Normalize and reconstruct
    printf("Normalizing and reconstructing...\n");
    normalize_accumulator(acc);
    Image* result = reconstruct_merged_image(acc);
    
    if (result) {
        printf("Saving result images...\n");
        save_image_ppm(ref_img, "test_ref.ppm");
        save_image_ppm(target_img, "test_target.ppm");
        save_image_ppm(result, "test_merged.ppm");
        free_test_image(result);
    }
    
    // Cleanup
    free_test_image(ref_img);
    free_test_image(target_img);
    free_merge_accumulator(acc);
    
    printf("Test completed\n");
    return 0;
} 