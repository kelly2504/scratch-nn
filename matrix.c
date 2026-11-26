#include "matrix.h"
#include <stdlib.h>

Matrix create_matrix(int rows, int cols) {
    // Matrix mat;
    // mat.rows = rows;
    // mat.cols = cols;

    Matrix mat = {rows, cols, NULL};

    //allocate memmory for a single 1d array 
    //mat.data = (double*)malloc((long)rows * cols * sizeof(double));

    // calloc zeroes the memory and checks the multiplication for overflow
    mat.data = calloc((size_t)rows * (size_t)cols, sizeof(double));
    if (mat.data == NULL) {
        fprintf(stderr, "create_matrix: allocation of %dx%d failed\n", rows, cols);
        exit(EXIT_FAILURE);
    }
    return mat;

}

void set_matrix_value(Matrix *mat, int row, int col, double value) {
    mat->data[row * mat->cols + col] = value;
}

double get_matrix_value(Matrix *mat, int row, int col) {
    return mat->data[row * mat->cols + col];
}

//calculate dot product 
Matrix matrix_multiply(Matrix *mat1, Matrix *mat2) {
    if (mat1->cols != mat2->rows) {
        fprintf(stderr, "matrix multiplication: shape mismatch %dx%d · %dx%d\n",
                mat1->rows, mat1->cols, mat2->rows, mat2->cols);
        exit(EXIT_FAILURE);
    }
    Matrix mat = create_matrix(mat1->rows, mat2->cols);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            for (int k = 0; k < mat1->cols; k++) {
                mat.data[i * mat.cols + j] += mat1->data[i * mat1->cols + k] * mat2->data[k * mat2->cols +j];
            }
        }
    }

    return mat;
}

Matrix dot_product(Matrix *mat1, Matrix *mat2) {
     if (mat1->cols != mat2->rows) {
        fprintf(stderr, "dot product: shape mismatch %dx%d · %dx%d\n",
                mat1->rows, mat1->cols, mat2->rows, mat2->cols);
        exit(EXIT_FAILURE);
    }
    Matrix mat = create_matrix(mat1->rows, mat2->cols);

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            for (int k = 0; k < mat1->cols; k++) {
                mat.data[i * mat.cols + j] += mat1->data[i * mat1->cols + k] * mat2->data[k * mat2->cols +j];
            }
        }
    }

    return mat;
}

Matrix add_matrices(Matrix mat1, Matrix mat2) {
    if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
        fprintf(stderr, "add_matrices: shape mismatch %dx%d + %dx%d\n",
                mat1.rows, mat1.cols, mat2.rows, mat2.cols);
        exit(EXIT_FAILURE);
    }

    Matrix mat = create_matrix(mat1.rows, mat1.cols);

    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            int pos = i * mat1.cols + j;
            mat.data[pos] = mat1.data[pos] + mat2.data[pos];
        }
    }

    return mat;
}

Matrix subtract_matrices(Matrix mat1, Matrix mat2) {
     if (mat1.rows != mat2.rows || mat1.cols != mat2.cols) {
        fprintf(stderr, "add_matrices: shape mismatch %dx%d + %dx%d\n",
                mat1.rows, mat1.cols, mat2.rows, mat2.cols);
        exit(EXIT_FAILURE);
    }

    Matrix mat = create_matrix(mat1.rows, mat1.cols);
    for (int i = 0; i < mat1.rows; i++) {
        for (int j = 0; j < mat1.cols; j++) {
            int pos = i * mat1.cols + j;
            mat.data[pos] = mat1.data[pos] - mat2.data[pos];
        }
    }

    return mat;
}

Matrix multiply_matrix_with_scaler(double num, Matrix mat2) {
    for (int i = 0; i < mat2.rows; i++) {
        for (int j = 0; j < mat2.cols; j++) {
            mat2.data[i * mat2.cols + j] *= num;
        }
    }
    return mat2;
}

Matrix matrix_transpose(Matrix mat) {
    Matrix mat2 = create_matrix(mat.cols, mat.rows);

    // for (int i = 0; i < mat.cols; i++ ) {
    //     for (int j = 0; j < mat.rows; j++) {
    //         mat2.data[i * mat2.cols + j] = mat.data[j * mat.cols + i];
    //     }
    // }

    for (int i = 0; i < mat.rows; i++ ) {
        for (int j = 0; j < mat.cols; j++) {
            mat2.data[j * mat2.cols + i] = mat.data[i * mat.cols + j];
        }
    }
    return mat2;
}

Matrix matrix_add_column_vector(Matrix mat, Matrix col) {
    if (col.rows != mat.rows || col.cols != 1) {
        fprintf(stderr, "add_column_vecotr: shape mismatch %dx%d + %dx%d\n",
                mat.rows, mat.cols, col.rows, col.cols);
        exit(EXIT_FAILURE);
    }
    Matrix out = create_matrix(mat.rows, mat.cols);
    
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            out.data[i * mat.cols + j] = mat.data[i * mat.cols + j] + col.data[i];
        }
    }

    return out;
}

Matrix matrix_add_scaler(Matrix mat, double z) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            mat.data[i * mat.cols + j] += z;
        }
    }
    return mat;
}

// Like np.sum(mat, axis, keepdims=True):
//   axis 0 -> sum down each column -> 1 x cols
//   axis 1 -> sum across each row  -> rows x 1
Matrix matrix_np_sum(Matrix *mat, int axis) {
    if (axis != 0 && axis != 1) {
        fprintf(stderr, "matrix_np_sum: axis must be 0 or 1, got %d\n", axis);
        exit(EXIT_FAILURE);
    }

    Matrix out = (axis == 0) ? create_matrix(1, mat->cols)
                             : create_matrix(mat->rows, 1);  // calloc -> starts at 0

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            double v = mat->data[i * mat->cols + j];
            if (axis == 0) {
                out.data[j] += v;   // column j total
            } else {
                out.data[i] += v;   // row i total
            }
        }
    }

    return out;
}
// Like A * B in numpy (Hadamard product): both matrices must have the same
// shape, and each output entry is the product of the entries at that position.
// Not the same as matrix_multiply / dot_product (A.dot(B)).
Matrix matrix_elementwise_multiply(Matrix *mat1, Matrix *mat2) {
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) {
        fprintf(stderr, "matrix_elementwise_multiply: shape mismatch %dx%d * %dx%d\n",
                mat1->rows, mat1->cols, mat2->rows, mat2->cols);
        exit(EXIT_FAILURE);
    }

    Matrix out = create_matrix(mat1->rows, mat1->cols);
    int n = mat1->rows * mat1->cols;
    for (int i = 0; i < n; i++) {
        out.data[i] = mat1->data[i] * mat2->data[i];
    }

    return out;
}
