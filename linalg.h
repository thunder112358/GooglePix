#ifndef LINALG_H
#define LINALG_H

#include <stdbool.h>
#include "utils.h"
#include "common.h"

// Vector structure for 2D vectors
typedef struct {
    float x;
    float y;
} Vector2D;

// Function declarations for linear algebra operations
void solve_2x2(const Matrix2x2* A, const Vector2D* b, Vector2D* x);
void matrix2x2_inverse(const Matrix2x2* A, Matrix2x2* inv);
float matrix2x2_determinant(const Matrix2x2* A);

// Interpolation functions
float bilinear_interpolate(const float* image, int height, int width, float y, float x);
void bilinear_interpolate_rgb(const Image* image, float y, float x, float* rgb);

// Gradient computation
void compute_gradients(const Image* image, float* dx, float* dy);
void compute_gradient_x(const float* image, float* dx, int height, int width);
void compute_gradient_y(const float* image, float* dy, int height, int width);

// Convolution operations
void convolve_2d(const float* input, float* output, const float* kernel,
                 int height, int width, int kernel_size);
void gaussian_blur(float* image, int height, int width, float sigma);

// Helper functions
float gaussian_kernel_1d(float x, float sigma);
void create_gaussian_kernel(float* kernel, int size, float sigma);

// Eigenvalue/eigenvector computation
void get_real_polyroots_2(float a, float b, float c, float* roots);
void get_eigen_val_2x2(const Matrix2x2* M, float* eigenvals);
void get_eigen_vect_2x2(const Matrix2x2* M, const float* eigenvals, 
                        Vector2D* e1, Vector2D* e2);
void get_eigen_elmts_2x2(const Matrix2x2* M, float* eigenvals, 
                        Vector2D* e1, Vector2D* e2);
float quad_mat_prod(const Matrix2x2* A, float x1, float x2);
void interpolate_cov(const Matrix2x2 covs[2][2], const Vector2D* center_pos, 
                    Matrix2x2* interpolated_cov);

#endif // LINALG_H 