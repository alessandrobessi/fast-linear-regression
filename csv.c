#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_matrix.h>

void save_matrix_to_csv(const gsl_matrix *Z, const int num_features, const int num_examples, const char file_name[])
{
    FILE *fp = fopen(file_name, "w");
    if (fp == NULL)
    {
        printf("Can't open %s\n", file_name);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_examples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            fprintf(fp, "%g", gsl_matrix_get(Z, i, j));
            if (j < num_features - 1)
                fprintf(fp, ",");
        }
        if (i < num_examples - 1)
            fprintf(fp, "\n");
    }
    fclose(fp);
    free(fp);

    printf("Data saved to %s\n", file_name);
}

void get_matrix_dims(const char path[], int *num_features, int *num_examples)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Can't read %s\n", path);
        exit(EXIT_FAILURE);
    }

    const size_t LINE_SIZE = 1024 * 1024;
    char *line = malloc(LINE_SIZE);
    char buffer[LINE_SIZE];
    char *token;

    // count lines, count tokens
    int num_lines = 0;
    int num_tokens = 0;
    while ((fgets(line, LINE_SIZE, fp)) != NULL)
    {
        strcpy(buffer, line);
        token = strtok(buffer, ",");

        while (token != NULL)
        {
            if (num_lines == 0)
                num_tokens++;
            token = strtok(NULL, ",");
        }
        num_lines++;
    }
    fclose(fp);
    free(fp);

    printf("%s contains %d features and %d examples\n", path, num_tokens, num_lines);

    *num_features = num_tokens;
    *num_examples = num_lines;
}

void load_matrix_from_csv(const char path[], gsl_matrix *Q)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Can't read %s\n", path);
        exit(EXIT_FAILURE);
    }

    printf("Reading %s\n", path);

    const size_t LINE_SIZE = 1024 * 1024;
    char *line = malloc(LINE_SIZE);
    char buffer[LINE_SIZE];
    char *token;

    int count_line = 0;
    int count_token = 0;
    while ((fgets(line, LINE_SIZE, fp)) != NULL)
    {
        strcpy(buffer, line);
        token = strtok(buffer, ",");
        count_token = 0;
        while (token != NULL)
        {
            gsl_matrix_set(Q, count_line, count_token, atof(token));
            count_token++;
            token = strtok(NULL, ",");
        }
        count_line++;
    }

    fclose(fp);
    free(fp);
}
