#include "backward_prop.h"
#include "matrix.h"
#include <stdlib.h>

Matrix one_hot(const Matrix* labels, int num_classes) {
    int m = labels->rows * labels->cols;
    Matrix Y = create_matrix(num_classes, m); //already zero-fill
    for (int i = 0; i < m; i++) {
        int label = (int)labels->data[i];
        if (label < 0 || label >= num_classes) {
            fprintf(stderr, "one_hot: label %d at index %d out of range", label, i);
            exit(EXIT_FAILURE);
        }
        set_matrix_value(&Y, label, i, 1.0);
    }

    return Y;
}

// BackwardProp backward_propagation(ForwardProp *params, Matrix X, Matrix Y) {
//     int m = Y.cols * Y.rows;

//     BackwardProp backprop;

//     //how to turn Y into const int* labels?
//     Matrix one_hot_y = one_hot(&Y, m);

//     //matrix subtraction 
//     backprop.dz2 = subtract_matrices(params->a2, one_hot_y);

//     //transpose matrix with T 
//     Matrix transposed_a2 = matrix_transpose(params->a2);
//     backprop.dw2 = multiply_matrix_with_scaler((double) (1.0/m), dot_product(&backprop.dz2, &transposed_a2));
    

// }