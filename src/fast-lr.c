#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "csv.h"
#include "generate.h"
#include "fit.h"
#include "predict.h"

void print_usage()
{
    printf("Something went wrong...\n\n");
    printf("Usage\n-----\n");
    printf("fast-lr generate [num_features] [num_examples]\n");
    printf("fast-lr fit [X_train_csv_file] [y_train_csv_file] [--verbose] [--with-intercept]\n");
    printf("fast-lr predict [X_train_csv_file] [beta_csv_file] [--verbose] [--with-intercept]\n");
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "generate") != 0 && strcmp(argv[1], "fit") != 0 && strcmp(argv[1], "predict") != 0)
    {
        print_usage();
        exit(EXIT_FAILURE);
    }

    bool verbose = false;
    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "--verbose") == 0)
        {
            verbose = true;
        }
    }

    bool intercept = false;
    for (int i = 2; i < argc; i++)
    {
        if (strcmp(argv[i], "--with-intercept") == 0)
        {
            intercept = true;
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
            print_usage();
            exit(EXIT_FAILURE);
        }

        fit(argv[2], argv[3], verbose, intercept);
    }

    if (strcmp(argv[1], "predict") == 0)
    {

        if (argc < 3)
        {
            print_usage();
            exit(EXIT_FAILURE);
        }

        predict(argv[2], argv[3], verbose, intercept);
    }

    return 0;
}