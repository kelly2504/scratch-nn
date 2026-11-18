
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "forward_prop.h"

#define MAX_LINE_SIZE 8192


int read_mnist();
void read_file(FILE* filename);
Matrix data_matrix();
Matrix matrix_test();

int main(void) {
    // int read = read_mnist();

    //temporary matrix of 10 digits to predict 10 digits
    Matrix m1 = matrix_test();

    time_t time1 = time(NULL);
    matrix_transpose(m1);
    time_t time2 = time(NULL);

    printf("time taken: %f", difftime(time2,time1));
    return 0;
}

//make a 1d matrix with 10 digits 0 to 9
Matrix data_matrix() {
    Matrix mat = create_matrix(10, 1);
    for (int i = 0; i < 10; i++) {
        set_matrix_value(&mat, i, 0, i);
    }

    return mat;
}

Matrix matrix_test() {
    Matrix mat = create_matrix(100000, 100000);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            set_matrix_value(&mat, i, j, i*j);
        }
    }
    return mat;
}

//reading from the MNIST dataset 
int read_mnist() {

    const char* train_file = "./archive/mnist_train.csv";
    const char* test_file = "./archive/mnist_test.csv";

    int is_ok = EXIT_FAILURE;


    FILE* train_fp = fopen(train_file, "r");
    FILE* test_fp = fopen(test_file, "r");

    if (!train_fp) {
        perror("Failed to open file");
        return is_ok;
    }

    if (!test_fp) {
        perror("Failed to open file");
        return is_ok;
    }


    printf("READING TRAINING FILE");
    read_file(train_fp);

    //read testing file 
    printf("READING TESTING FILE");
    read_file(test_fp);
   
    fclose(train_fp);
    fclose(test_fp);

    return EXIT_SUCCESS;
}

void read_file(FILE* filename) {
char buffer[MAX_LINE_SIZE];
int row = 0;

    // read testing file
    while (fgets(buffer, MAX_LINE_SIZE, filename) != NULL) {
        row++;

        if (row == 1 ) {
            continue;
        }

        printf("\r\033[KReading row %d:\n", row);

        //tokenize line with ',' as a delimiter
        char *token = strtok(buffer, ",");
        int column = 0;

        while (token != 0) {
            token[strcspn(token, "\r\n")] = 0;

            printf("  \r\033[KColumn %d: %s\n", column, token);

            //move to the next token 
            token = strtok(NULL, ",");
            column++;

        }
        printf("\n");

    } 
}

