#include "forward_prop.h"
#include "matrix.h"

//initialise bias and weight 
void init_parameters(ForwardProp* params, int matrix_size, int target_size) {
    srand(time(NULL));
    //populate parameter matrices
    params->w1 = create_matrix(target_size, matrix_size);
    for (int i = 0; i < target_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            //where max = 784 and min = 10
            double w1 = (rand() % (matrix_size - target_size + 1)) + 10 - 0.5;
            set_matrix_value(&params->w1, i, j, w1);
        }
    }

    params->b1 = create_matrix(target_size, 1);
    for (int i = 0; i < target_size; i++) {
        double b1 = (rand() % target_size) - 1 - 0.5;
        set_matrix_value(&params->b1, i, 0, b1);
    }

    params->w2 = create_matrix(target_size, target_size);
    for (int i = 0; i < target_size; i++) {
        for (int j = 0; j < target_size; j++) {
            double w2 = (rand() % 1) + target_size - 0.5;
            set_matrix_value(&params->w2, i, j, w2);
        }
    }

    params->b2 = create_matrix(target_size, 1);
    for (int i = 0; i < target_size; i++) {
        double b2 = (rand() % 1) + target_size - 0.5;
        set_matrix_value(&params->b2, i, 0, b2);
    }

}

//actual forward propagation code
void forward_propagation(ForwardProp *params, Matrix X, int input_size, int output_size) {
    params->z1 = matrix_add_column_vector(dot_product(&params->w1, &X), params->b1);
    params->a1 = relu_matrix(&params->z1);
    params->z2 = matrix_add_column_vector(dot_product(&params->w2, &params->a1), params->b2);

    params->a2 = create_matrix(output_size, output_size);
    params->a2 = softmax(params->a1, params->a2, input_size, output_size);
}

//returns x if x is greater than 0 else 0 
double RELU(double x) {
    return (x > 0.0) ? x : 0.0;
}

Matrix relu_matrix(Matrix* mat) {
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = RELU(mat->data[i]);
    }
    return *mat;
}


Matrix softmax(Matrix input, Matrix output, int input_size, int output_size) {
    double max = input.data[0];
    double sum = 0.0f;

    // 1. Find the maximum value in the input array for numerical stability
    for (int i = 0; i < input_size; i++) {
        max = (input.data[i] > max) ? input.data[i] : max;
    }

    // 2. Compute exponentials and their sum
    for (int i = 0; i < output_size; i++) {
        output.data[i] = expf(input.data[i] - max);
        sum += output.data[i];
    }

    // 3. Divide each exponential by the sum to get probabilities
    for (int i = 0; i < output_size; i++) {
        output.data[i] /= sum;
    }

    return output;

}