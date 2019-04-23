#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics_double.h>
#include <stdbool.h>
#include "csv.h"

void predict(const char x_file_name[], const char beta_file_name[], const bool verbose, const bool intercept,
             const bool normalize) {
    int num_features = 0;
    int num_examples = 0;

    int *ptr_num_features = &num_features;
    int *ptr_num_examples = &num_examples;

    get_matrix_dims(x_file_name, ptr_num_features, ptr_num_examples);

    if (intercept)
        num_features++;

    gsl_matrix *X = gsl_matrix_alloc(num_examples, num_features);
    load_matrix_from_csv(x_file_name, X, intercept);

    if (normalize) {
        gsl_vector *col = gsl_vector_alloc(num_examples);
        double data[num_examples];
        double mean, sd;
        for (int p = 0; p < num_features; p++) {
            for (int k = 0; k < num_examples; k++) {
                data[k] = gsl_matrix_get(X, k, p);
            }
            mean = gsl_stats_mean(data, 1, num_examples);
            sd = gsl_stats_sd(data, 1, num_examples);
            gsl_matrix_get_col(col, X, p);
            gsl_vector_add_constant(col, mean);
            gsl_vector_scale(col, 1 / sd);
            gsl_matrix_set_col(X, p, col);
        }
    }

    gsl_matrix *beta = gsl_matrix_alloc(num_features, 1);
    load_matrix_from_csv(beta_file_name, beta, false);

    gsl_matrix *y_hat = gsl_matrix_alloc(num_examples, 1);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, X, beta, 0.0, y_hat);

    // SHOW Y_HAT
    if (verbose) {
        printf("y_hat:\n");
        for (int i = 0; i < num_examples; i++) {
            printf("y_hat[%d] = %.3g\n", i, gsl_matrix_get(y_hat, i, 0));
        }
        printf("\n");
    }

    save_matrix_to_csv(y_hat, 1, num_examples, "y_hat.csv");
}