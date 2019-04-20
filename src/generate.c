#include <gsl/gsl_matrix.h>
#include <time.h>
#include <stdbool.h>
#include "csv.h"

void generate_data(int num_features, int num_examples)
{
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

    save_matrix_to_csv(X, num_features, num_examples, "X_train.csv");
    save_matrix_to_csv(y, 1, num_examples, "y_train.csv");
    save_matrix_to_csv(params, 1, num_features, "true_betas.csv");

    gsl_matrix_free(params);
    gsl_matrix_free(X);
    gsl_matrix_free(y);
}