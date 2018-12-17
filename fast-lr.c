#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "csv.h"
#include "generate.h"
#include "estimate.h"
#include "predict.h"

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("See usage!\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "generate") != 0 && strcmp(argv[1], "fit") != 0 && strcmp(argv[1], "predict") != 0)
    {
        printf("See usage!\n");
        exit(EXIT_FAILURE);
    }

    bool verbose = false;
    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "--verbose") == 0)
        {
            verbose = true;
            if (verbose)
                printf("Verbose mode\n");
        }
    }

    if (strcmp(argv[1], "generate") == 0)
    {
        int num_features = atoi(argv[2]);
        int num_examples = atoi(argv[3]);

        printf("Generating sample data with %d features and %d examples.\n", num_features, num_examples);
        generate_data(num_features, num_examples);
        printf("Done! You can use X_train.csv and y_train.csv\n");
        exit(EXIT_SUCCESS);
    }

    if (strcmp(argv[1], "fit") == 0)
    {

        if (argc < 4)
        {
            printf("Error. You must provide a source csv file for examples and a source csv file for labels.\n");
            exit(EXIT_FAILURE);
        }

        bool with_intercept = false;
        for (int i = 2; i < argc; i++)
        {
            if (strcmp(argv[i], "--with-intercept") == 0)
            {
                with_intercept = true;
                if (with_intercept)
                    printf("Fitting with intercept\n");
            }
        }

        fit(argv[2], argv[3]);
    }

    if (strcmp(argv[1], "predict") == 0)
    {

        if (argc < 3)
        {
            printf("Error. You must provide a source csv file for examples and a source csv file for labels.\n");
            exit(EXIT_FAILURE);
        }

        predict(argv[2], argv[3]);
    }

    return 0;
}