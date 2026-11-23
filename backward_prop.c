#include "backward_prop.h"
#include "forward_prop.h"
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

Matrix derive_relu(Matrix mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            mat.data[i * mat.cols + j] = (mat.data[i * mat.cols +j] > 0 ? 1.0 : 0.0 );
        }
    }
    return mat;
}

BackwardProp backward_propagation(ForwardProp *params, Matrix X, Matrix Y) {
    //m = y.size
    int m = Y.cols * Y.rows;

    BackwardProp backprop;

    //how to turn Y into const int* labels?
    Matrix one_hot_y = one_hot(&Y, m);


    //MARK: First layer
    //matrix subtraction 
    //dz2 = A2 - one_hot_y
    backprop.dz2 = subtract_matrices(params->a2, one_hot_y);

    //transpose matrix with T 
    //dw2 = (1/m) * dz2.dot(A1.T)
    Matrix transposed_a1 = matrix_transpose(params->a1);
    backprop.dw2 = multiply_matrix_with_scaler((double) (1.0/m), dot_product(&backprop.dz2, &transposed_a1));

    //db2 = (1/m) * np.sum(dz2, 1)
    backprop.db2 = multiply_matrix_with_scaler(1.0 / m, matrix_np_sum(&backprop.dz2, 1));


    //MARK: Second layer 
    //dz1 = w2.T.dot(dz2) * deriv_Relu(z1)
    Matrix transposed_w2 = matrix_transpose(params->w2);
    Matrix dot_w2T_dz2 = dot_product(&transposed_w2, &backprop.dz2);
    Matrix derived_relu_z1 = derive_relu(params->z1);
    backprop.dz1 = matrix_multiply(&dot_w2T_dz2, &derived_relu_z1);

    //dw1 = (1/m) * dz1.dot(X.T)
    Matrix transposed_X = matrix_transpose(X);
    backprop.dw1 = multiply_matrix_with_scaler((1.0/m), dot_product(&backprop.dz1, &transposed_X));

    //db1 = (1/m) * np.sum(dz1, 1)
    backprop.db1 = multiply_matrix_with_scaler(1.0 / m, matrix_np_sum(&backprop.dz1, 1));

    return backprop;

}

//alpha here representing learning rate 

ForwardProp update_params(ForwardProp *params, BackwardProp *derivatives, double alpha) {
    //w1 = w1 - alpha * dw1
    params->w1 = subtract_matrices(params->w1, multiply_matrix_with_scaler(alpha, derivatives->dw1));

    //b1 = b1 - alpha * db1
    params->b1 = subtract_matrices(params->b1, multiply_matrix_with_scaler(alpha, derivatives->db1));

    //w2 = w2 - alpha * dw2
    params->w2 = subtract_matrices(params->w2, multiply_matrix_with_scaler(alpha, derivatives->dw2));

    //b2 = b2 - alpha * db2
    params->b2 = subtract_matrices(params->b2, multiply_matrix_with_scaler(alpha, derivatives->db2));
    return *params;
}