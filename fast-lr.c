#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include "csv.h"

void generate_data(int num_features, int num_examples)
{
    // GENERATE DATA
    srand(time(NULL));
    double range = 1.0 * RAND_MAX;

    gsl_matrix *X = gsl_matrix_alloc(num_examples, num_features);
    gsl_matrix *y = gsl_matrix_alloc(num_examples, 1);

    gsl_matrix *params = gsl_matrix_alloc(num_features, 1);

    for (int i = 0; i < num_features; i++)
    {
        gsl_matrix_set(params, i, 0, rand() / range);
    }

    for (int i = 0; i < num_examples; i++)
        for (int j = 0; j < num_features; j++)
            gsl_matrix_set(X, i, j, rand() / range);

    for (int i = 0; i < num_examples; i++)
    {
        double value = 0;
        for (int j = 0; j < num_features; j++)
        {
            value += gsl_matrix_get(X, i, j) * gsl_matrix_get(params, j, 0);
        }
        gsl_matrix_set(y, i, 0, value + rand() / (range * 1000));
    }

    // CREATE TRAIN MATRIX
    gsl_matrix *Z = gsl_matrix_alloc(num_examples, num_features + 1);
    for (int i = 0; i < num_examples; i++)
    {
        gsl_matrix_set(Z, i, 0, gsl_matrix_get(y, i, 0));
        for (int j = 0; j < num_features; j++)
        {
            gsl_matrix_set(Z, i, j + 1, gsl_matrix_get(X, i, j));
        }
    }

    // SAVE MATRIX TO CSV
    save_matrix_to_csv(Z, num_features, num_examples);
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("See usage!\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "generate") != 0 && strcmp(argv[1], "estimate") != 0)
    {
        printf("See usage!\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "generate") == 0)
    {
        int num_features = atoi(argv[2]);
        int num_examples = atoi(argv[3]);

        printf("Generating sample data with %d features and %d examples.\n", num_features, num_examples);
        generate_data(num_features, num_examples);
        printf("Done! You can use sample_train.csv\n");
        exit(EXIT_SUCCESS);
    }

    if (strcmp(argv[1], "estimate") == 0)
    {

        if (argc < 3)
        {
            printf("Error. You must provide a source csv file.\n");
            exit(EXIT_FAILURE);
        }

        int num_features = 0;
        int num_examples = 0;

        int *ptr_num_features = &num_features;
        int *ptr_num_examples = &num_examples;

        get_matrix_dims(argv[2], ptr_num_features, ptr_num_examples);

        gsl_matrix *Q = gsl_matrix_alloc(num_examples, num_features + 1);
        load_matrix_from_csv(argv[2], Q);

        gsl_matrix *X = gsl_matrix_alloc(num_examples, num_features);
        gsl_matrix *y = gsl_matrix_alloc(num_examples, 1);

        for (int i = 0; i < num_examples; i++)
        {
            gsl_matrix_set(y, i, 0, gsl_matrix_get(Q, i, 0));

            for (int j = 0; j < num_features; j++)
            {
                gsl_matrix_set(X, i, j, gsl_matrix_get(Q, i, j + 1));
            }
        }

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

        // SHOW RESULTS
        printf("beta estimates:\n");
        for (int i = 0; i < num_features; i++)
            printf("est beta[%d] = %.3g\n", i, gsl_matrix_get(beta, i, 0));
        printf("\n");
    }

    return 0;
}