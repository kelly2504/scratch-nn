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

// Returns a NEW matrix of 1s and 0s (1 where mat > 0); the input is left unchanged.
// Modifying in place would wipe out params->z1.
Matrix derive_relu(const Matrix* mat) {
    Matrix out = create_matrix(mat->rows, mat->cols);
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        out.data[i] = (mat->data[i] > 0) ? 1.0 : 0.0;
    }
    return out;
}

// Note: multiply_matrix_with_scaler scales its argument in place and returns
// that same buffer, so wrapping a freshly created matrix in it doesn't leak.
BackwardProp backward_propagation(ForwardProp *params, Matrix X, Matrix Y) {
    //m = y.size
    int m = Y.cols * Y.rows;

    BackwardProp backprop;

    Matrix one_hot_y = one_hot(&Y, 10);


    //MARK: Output layer
    //dz2 = A2 - one_hot_y
    backprop.dz2 = subtract_matrices(params->a2, one_hot_y);
    free(one_hot_y.data);

    //dw2 = (1/m) * dz2.dot(A1.T)
    Matrix transposed_a1 = matrix_transpose(params->a1);
    backprop.dw2 = multiply_matrix_with_scaler(1.0 / m, dot_product(&backprop.dz2, &transposed_a1));
    free(transposed_a1.data);

    //db2 = (1/m) * np.sum(dz2, 1)
    backprop.db2 = multiply_matrix_with_scaler(1.0 / m, matrix_np_sum(&backprop.dz2, 1));


    //MARK: Hidden layer
    //dz1 = w2.T.dot(dz2) * deriv_Relu(z1)
    Matrix transposed_w2 = matrix_transpose(params->w2);
    Matrix dot_w2T_dz2 = dot_product(&transposed_w2, &backprop.dz2);
    Matrix derived_relu_z1 = derive_relu(&params->z1);
    backprop.dz1 = matrix_elementwise_multiply(&dot_w2T_dz2, &derived_relu_z1);
    free(transposed_w2.data);
    free(dot_w2T_dz2.data);
    free(derived_relu_z1.data);

    //dw1 = (1/m) * dz1.dot(X.T)
    Matrix transposed_X = matrix_transpose(X);
    backprop.dw1 = multiply_matrix_with_scaler(1.0 / m, dot_product(&backprop.dz1, &transposed_X));
    free(transposed_X.data);

    //db1 = (1/m) * np.sum(dz1, 1)
    backprop.db1 = multiply_matrix_with_scaler(1.0 / m, matrix_np_sum(&backprop.dz1, 1));

    return backprop;
}

// param = param - alpha * grad, written straight into param's existing memory.
// No new matrices are created, so nothing to free, and grad is left unchanged.
static void step(Matrix* param, const Matrix* grad, double alpha) {
    if (param->rows != grad->rows || param->cols != grad->cols) {
        fprintf(stderr, "update_params: shape mismatch %dx%d - %dx%d\n",
                param->rows, param->cols, grad->rows, grad->cols);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < param->rows * param->cols; i++) {
        param->data[i] -= alpha * grad->data[i];
    }
}

//alpha here representing learning rate
ForwardProp* update_params(ForwardProp *params, BackwardProp *derivatives, double alpha) {
    step(&params->w1, &derivatives->dw1, alpha);  //w1 = w1 - alpha * dw1
    step(&params->b1, &derivatives->db1, alpha);  //b1 = b1 - alpha * db1
    step(&params->w2, &derivatives->dw2, alpha);  //w2 = w2 - alpha * dw2
    step(&params->b2, &derivatives->db2, alpha);  //b2 = b2 - alpha * db2
    return params;
}

// Frees every matrix in the struct
void free_backward_prop(BackwardProp* backprop) {
    free(backprop->dw1.data); backprop->dw1.data = NULL;
    free(backprop->dw2.data); backprop->dw2.data = NULL;
    free(backprop->db1.data); backprop->db1.data = NULL;
    free(backprop->db2.data); backprop->db2.data = NULL;
    free(backprop->dz1.data); backprop->dz1.data = NULL;
    free(backprop->dz2.data); backprop->dz2.data = NULL;
}
