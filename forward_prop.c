#include "forward_prop.h"
#include "matrix.h"

// Random value in [-0.5, 0.5], like np.random.rand(...) - 0.5
static double rand_weight(void) {
    return ((double)rand() / RAND_MAX) - 0.5;
}

//initialise bias and weight
//  matrix_size: number of input features (784 for MNIST)
//  target_size: number of neurons in the hidden and output layers (10)
ForwardProp* init_parameters(ForwardProp* params, int matrix_size, int target_size) {
    srand(time(NULL));

    // Small random values centred on 0. Large or identical weights make
    // softmax saturate and every output neuron learn the same thing.
    params->w1 = create_matrix(target_size, matrix_size);
    for (int i = 0; i < target_size * matrix_size; i++) params->w1.data[i] = rand_weight();

    params->b1 = create_matrix(target_size, 1);
    for (int i = 0; i < target_size; i++) params->b1.data[i] = rand_weight();

    params->w2 = create_matrix(target_size, target_size);
    for (int i = 0; i < target_size * target_size; i++) params->w2.data[i] = rand_weight();

    params->b2 = create_matrix(target_size, 1);
    for (int i = 0; i < target_size; i++) params->b2.data[i] = rand_weight();

    // No activations yet. NULL lets forward_propagation free() them safely
    // on the first pass (free(NULL) does nothing).
    params->z1 = (Matrix){0, 0, NULL};
    params->a1 = (Matrix){0, 0, NULL};
    params->z2 = (Matrix){0, 0, NULL};
    params->a2 = (Matrix){0, 0, NULL};

    return params;
}

//actual forward propagation code
//  X: (input features x m samples), one column per sample
//  Shapes produced: z1, a1 = (hidden x m); z2, a2 = (classes x m)
ForwardProp* forward_propagation(ForwardProp *params, Matrix X, int input_size, int output_size) {
    (void)input_size;   // shapes now come from the matrices themselves
    (void)output_size;

    // Free the previous pass's activations before overwriting them
    free(params->z1.data);
    free(params->a1.data);
    free(params->z2.data);
    free(params->a2.data);

    //z1 = w1.dot(X) + b1
    Matrix w1_x = dot_product(&params->w1, &X);
    params->z1 = matrix_add_column_vector(w1_x, params->b1);
    free(w1_x.data);

    //a1 = ReLU(z1)  (new matrix, so z1 keeps its pre-activation values)
    params->a1 = relu_matrix(&params->z1);

    //z2 = w2.dot(a1) + b2
    Matrix w2_a1 = dot_product(&params->w2, &params->a1);
    params->z2 = matrix_add_column_vector(w2_a1, params->b2);
    free(w2_a1.data);

    //a2 = softmax(z2), applied to each column (each sample) separately
    params->a2 = softmax(params->z2);

    return params;
}

// Frees every matrix in the struct (not the struct itself)
void free_forward_prop(ForwardProp* params) {
    free(params->w1.data); params->w1.data = NULL;
    free(params->b1.data); params->b1.data = NULL;
    free(params->w2.data); params->w2.data = NULL;
    free(params->b2.data); params->b2.data = NULL;
    free(params->z1.data); params->z1.data = NULL;
    free(params->a1.data); params->a1.data = NULL;
    free(params->z2.data); params->z2.data = NULL;
    free(params->a2.data); params->a2.data = NULL;
}

//returns x if x is greater than 0 else 0
double RELU(double x) {
    return (x > 0.0) ? x : 0.0;
}

// Returns a NEW matrix; the input is left unchanged.
Matrix relu_matrix(Matrix* mat) {
    Matrix out = create_matrix(mat->rows, mat->cols);
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        out.data[i] = RELU(mat->data[i]);
    }
    return out;
}

// Column-wise softmax: each column of the result sums to 1.
// Like np.exp(Z) / np.sum(np.exp(Z), axis=0), with the column max
// subtracted first so exp() can't overflow.
Matrix softmax(Matrix input) {
    Matrix output = create_matrix(input.rows, input.cols);

    for (int j = 0; j < input.cols; j++) {
        // 1. Find the maximum value in this column for numerical stability
        double max = input.data[j];
        for (int i = 1; i < input.rows; i++) {
            double v = input.data[i * input.cols + j];
            if (v > max) max = v;
        }

        // 2. Compute exponentials and their sum
        double sum = 0.0;
        for (int i = 0; i < input.rows; i++) {
            double e = exp(input.data[i * input.cols + j] - max);
            output.data[i * input.cols + j] = e;
            sum += e;
        }

        // 3. Divide each exponential by the sum to get probabilities
        for (int i = 0; i < input.rows; i++) {
            output.data[i * input.cols + j] /= sum;
        }
    }

    return output;
}
