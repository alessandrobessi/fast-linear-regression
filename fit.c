#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include "csv.h"

void fit(const char x_file_name[], const char y_file_name[])
{
    int num_features = 0;
    int num_examples = 0;

    int *ptr_num_features = &num_features;
    int *ptr_num_examples = &num_examples;

    get_matrix_dims(x_file_name, ptr_num_features, ptr_num_examples);

    gsl_matrix *X = gsl_matrix_alloc(num_examples, num_features);
    load_matrix_from_csv(x_file_name, X);

    gsl_matrix *y = gsl_matrix_alloc(num_examples, 1);
    load_matrix_from_csv(y_file_name, y);

    // DO THE MATH
    gsl_matrix *XT = gsl_matrix_alloc(num_features, num_examples);
    gsl_matrix *XTX = gsl_matrix_alloc(num_features, num_features);

    gsl_matrix_transpose_memcpy(XT, X);

    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, XT, X, 0.0, XTX);

    int s;
    gsl_permutation *p = gsl_permutation_alloc(num_features);
    gsl_linalg_LU_decomp(XTX, p, &s);
    gsl_matrix *inv = gsl_matrix_alloc(num_features, num_features);
    gsl_linalg_LU_invert(XTX, p, inv);

    gsl_matrix *invXT = gsl_matrix_alloc(num_features, num_examples);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, inv, XT, 0.0, invXT);

    gsl_matrix *beta = gsl_matrix_alloc(num_features, 1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, invXT, y, 0.0, beta);

    // COMPUTE ERRORS
    gsl_matrix *y_hat = gsl_matrix_alloc(num_examples, 1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, X, beta, 0.0, y_hat);

    gsl_matrix *u = gsl_matrix_alloc(num_examples, 1);
    gsl_matrix_memcpy(u, y);
    gsl_matrix_sub(u, y_hat);

    // COMPUTE SIGMA^2
    gsl_matrix *uT = gsl_matrix_alloc(1, num_examples);
    gsl_matrix *uTu = gsl_matrix_alloc(1, 1);
    gsl_matrix_transpose_memcpy(uT, u);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, uT, u, 0.0, uTu);

    double s2 = gsl_matrix_get(uTu, 0, 0) / (num_examples - num_features);

    printf("s2 = %g\n", s2);

    double correction_factor = (double)(num_examples - num_features) / num_examples;
    double sigma2 = correction_factor * s2;

    printf("sigma2 = %g\n", sigma2);

    gsl_matrix *vcov = gsl_matrix_alloc(num_features, num_features);
    gsl_matrix_memcpy(vcov, inv);
    gsl_matrix_scale(vcov, sigma2);

    // SHOW BETA ESTIMATES
    printf("beta estimates:\n");
    for (int i = 0; i < num_features; i++)
        printf("est beta[%d] = %.3g (%.3g)\n", i, gsl_matrix_get(beta, i, 0), gsl_matrix_get(vcov, i, i));
    printf("\n");

    printf("errors (y - y_hat):\n");
    for (int i = 0; i < num_examples; i++)
        printf("u[%d] = %.3g\n", i, gsl_matrix_get(u, i, 0));
    printf("\n");
}