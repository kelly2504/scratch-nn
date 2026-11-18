#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

typedef struct Matrix {
    int rows;
    int cols;
    double* data;

} Matrix;

Matrix create_matrix(int rows, int cols);
void set_matrix_value(Matrix* mat, int row, int col, double value);
double get_matrix_value(Matrix* mat, int row, int col);

Matrix matrix_multiply(Matrix* mat1, Matrix* mat2);
Matrix dot_product(Matrix* mat1, Matrix* mat2);
Matrix add_matrices(Matrix mat1, Matrix mat2);
Matrix subtract_matrices(Matrix mat1, Matrix mat2);
Matrix multiply_matrix_with_scaler(double scalar, Matrix mat2);
Matrix matrix_transpose(Matrix mat);
Matrix matrix_add_column_vector(Matrix mat, Matrix col);

#endif