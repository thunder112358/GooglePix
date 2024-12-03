#ifndef KERNELS_H
#define KERNELS_H

#include "utils.h"

// Kernel types
typedef enum {
    KERNEL_HANDHELD,
    KERNEL_ISO
} KernelType;

// Kernel parameters
typedef struct {
    KernelType type;       // Kernel type (handheld or iso)
    float k_detail;        // Detail preservation parameter
    float k_denoise;       // Denoising parameter
    float d_tr;           // Threshold for detail preservation
    float d_th;           // Threshold for high-frequency content
    float k_shrink;       // Kernel shrinkage parameter
    float k_stretch;      // Kernel stretching parameter
    int window_size;      // Size of the kernel window
} KernelParams;

// Steerable kernel structure
typedef struct {
    float* weights;       // Kernel weights
    float* orientations;  // Kernel orientations
    int size;            // Kernel size
    int count;           // Number of kernels
} SteerableKernels;

// Function declarations
SteerableKernels* estimate_kernels(const Image* image, const KernelParams* params);
void free_steerable_kernels(SteerableKernels* kernels);

// Helper functions
void compute_structure_tensor(const Image* image, float* Ixx, float* Ixy, float* Iyy);
void estimate_local_orientation(const float* Ixx, const float* Ixy, const float* Iyy,
                              float* magnitudes, float* orientations,
                              int height, int width);
void generate_steerable_basis(float* basis, int size, float orientation);
float compute_anisotropy(float lambda1, float lambda2);

#endif // KERNELS_H 