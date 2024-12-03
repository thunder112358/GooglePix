#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libraw/libraw.h>
#include <tiffio.h>
#include "utils_dng.h"

// Default paths for external tools
#define EXIFTOOL_PATH "exiftool"
#define DNG_VALIDATE_PATH "dng_validate"

// Create DNG metadata structure
DNGMetadata* create_dng_metadata(void) {
    DNGMetadata* metadata = (DNGMetadata*)calloc(1, sizeof(DNGMetadata));
    if (!metadata) return NULL;

    metadata->color_matrix1 = (ColorMatrix*)malloc(sizeof(ColorMatrix));
    metadata->color_matrix2 = (ColorMatrix*)malloc(sizeof(ColorMatrix));
    metadata->as_shot_neutral = (float*)malloc(3 * sizeof(float));

    if (!metadata->color_matrix1 || !metadata->color_matrix2 || !metadata->as_shot_neutral) {
        free_dng_metadata(metadata);
        return NULL;
    }

    return metadata;
}

// Free DNG metadata
void free_dng_metadata(DNGMetadata* metadata) {
    if (metadata) {
        free(metadata->color_matrix1);
        free(metadata->color_matrix2);
        free(metadata->as_shot_neutral);
        free(metadata);
    }
}

// Load DNG file using LibRaw
Image* load_dng(const char* filename, DNGMetadata* metadata) {
    libraw_data_t* raw = libraw_init(0);
    if (!raw) return NULL;

    if (libraw_open_file(raw, filename) != LIBRAW_SUCCESS) {
        libraw_close(raw);
        return NULL;
    }

    // Extract metadata if requested
    if (metadata) {
        metadata->width = raw->sizes.width;
        metadata->height = raw->sizes.height;
        metadata->bits_per_sample = raw->color.maximum > 255 ? 16 : 8;
        metadata->samples_per_pixel = 3;  // RGB
        metadata->photometric = PHOTO_CFA;
        metadata->compression = 7;  // JPEG lossless
        metadata->rows_per_strip = metadata->height;
        metadata->planar_config = 1;  // Chunky

        // Copy color matrices
        memcpy(metadata->color_matrix1->matrix, raw->color.cam_xyz, sizeof(float) * 9);
        memcpy(metadata->color_matrix2->matrix, raw->color.cam_xyz, sizeof(float) * 9);

        // Copy white balance
        for (int i = 0; i < 3; i++) {
            metadata->as_shot_neutral[i] = 1.0f / raw->color.cam_mul[i];
        }

        metadata->baseline_exposure = 0.0f;
        metadata->calibration_illuminant1 = 21;  // D65
        metadata->calibration_illuminant2 = 21;  // D65
    }

    // Unpack raw data
    if (libraw_unpack(raw) != LIBRAW_SUCCESS) {
        libraw_close(raw);
        return NULL;
    }

    // Convert to image
    libraw_processed_image_t* processed = libraw_dcraw_make_mem_image(raw);
    if (!processed) {
        libraw_close(raw);
        return NULL;
    }

    // Create image structure
    Image* image = (Image*)malloc(sizeof(Image));
    if (!image) {
        libraw_dcraw_clear_mem(processed);
        libraw_close(raw);
        return NULL;
    }

    image->height = processed->height;
    image->width = processed->width;
    image->channels = processed->colors;
    image->data = (float*)malloc(image->height * image->width * image->channels * sizeof(float));

    if (!image->data) {
        free(image);
        libraw_dcraw_clear_mem(processed);
        libraw_close(raw);
        return NULL;
    }

    // Convert to float and normalize
    float scale = 1.0f / (float)((1 << processed->bits) - 1);
    #pragma omp parallel for collapse(3)
    for (int y = 0; y < image->height; y++) {
        for (int x = 0; x < image->width; x++) {
            for (int c = 0; c < image->channels; c++) {
                int src_idx = (y * image->width + x) * image->channels + c;
                image->data[src_idx] = (float)processed->data[src_idx] * scale;
            }
        }
    }

    libraw_dcraw_clear_mem(processed);
    libraw_close(raw);
    return image;
}

// Save image as DNG
bool save_as_dng(const Image* image, const char* ref_dng_path, const char* output_path) {
    if (!image || !ref_dng_path || !output_path) return false;

    // Create TIFF file
    TIFF* tif = TIFFOpen(output_path, "w");
    if (!tif) return false;

    // Set basic TIFF tags
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, image->width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image->height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, image->channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 16);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, image->height);

    // Convert float data to 16-bit
    uint16_t* buffer = (uint16_t*)malloc(image->width * image->channels * sizeof(uint16_t));
    if (!buffer) {
        TIFFClose(tif);
        return false;
    }

    // Write image data
    for (int y = 0; y < image->height; y++) {
        for (int x = 0; x < image->width; x++) {
            for (int c = 0; c < image->channels; c++) {
                float val = image->data[(y * image->width + x) * image->channels + c];
                buffer[x * image->channels + c] = (uint16_t)(val * 65535.0f + 0.5f);
            }
        }
        TIFFWriteScanline(tif, buffer, y, 0);
    }

    free(buffer);
    TIFFClose(tif);

    // Copy DNG tags from reference file using exiftool
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "%s -n -overwrite_original -tagsfromfile %s "
        "\"-IFD0:AnalogBalance\" "
        "\"-IFD0:ColorMatrix1\" \"-IFD0:ColorMatrix2\" "
        "\"-IFD0:CameraCalibration1\" \"-IFD0:CameraCalibration2\" "
        "\"-IFD0:AsShotNeutral\" \"-IFD0:BaselineExposure\" "
        "\"-IFD0:CalibrationIlluminant1\" \"-IFD0:CalibrationIlluminant2\" "
        "\"-IFD0:ForwardMatrix1\" \"-IFD0:ForwardMatrix2\" "
        "\"%s\"",
        EXIFTOOL_PATH, ref_dng_path, output_path);

    if (system(cmd) != 0) {
        return false;
    }

    // Validate DNG file
    snprintf(cmd, sizeof(cmd), "%s -16 -dng \"%s\"",
             DNG_VALIDATE_PATH, output_path);
    
    return system(cmd) == 0;
}

// Check if photometric interpretation is supported
bool is_supported_photometric(int photometric) {
    switch (photometric) {
        case PHOTO_BLACK_IS_ZERO:
        case PHOTO_CFA:
            return true;
        default:
            return false;
    }
}

// Copy DNG tags from one file to another
void copy_dng_tags(const char* src_path, const char* dst_path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "%s -n -overwrite_original -tagsfromfile %s "
        "\"-all:all\" "
        "\"%s\"",
        EXIFTOOL_PATH, src_path, dst_path);
    
    system(cmd);
}

// Run DNG validation
bool run_dng_validate(const char* filename) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s -16 -dng \"%s\"",
             DNG_VALIDATE_PATH, filename);
    
    return system(cmd) == 0;
}

// Convert file to DNG format
bool convert_to_dng(const char* input_path, const char* output_path, const DNGMetadata* metadata) {
    if (!metadata) return false;

    // Create TIFF file with DNG tags
    TIFF* tif = TIFFOpen(output_path, "w");
    if (!tif) return false;

    // Set DNG tags
    TIFFSetField(tif, TIFFTAG_SUBFILETYPE, 0);
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, metadata->width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, metadata->height);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, metadata->bits_per_sample);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, metadata->compression);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, metadata->photometric);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, metadata->samples_per_pixel);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, metadata->rows_per_strip);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, metadata->planar_config);

    // Set color matrices
    if (metadata->color_matrix1) {
        TIFFSetField(tif, TIFFTAG_COLORMATRIX1, 9, metadata->color_matrix1->matrix);
    }
    if (metadata->color_matrix2) {
        TIFFSetField(tif, TIFFTAG_COLORMATRIX2, 9, metadata->color_matrix2->matrix);
    }

    // Set white balance
    if (metadata->as_shot_neutral) {
        TIFFSetField(tif, TIFFTAG_ASSHOTNEUTRAL, 3, metadata->as_shot_neutral);
    }

    TIFFClose(tif);

    // Copy image data from input
    copy_dng_tags(input_path, output_path);

    return run_dng_validate(output_path);
} 