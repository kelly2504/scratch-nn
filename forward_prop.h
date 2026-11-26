#ifndef FORWARD_PROP_H
#define FORWARD_PROP_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "matrix.h"

typedef struct ForwardProp {
    Matrix w1;
    Matrix w2;
    Matrix b1;
    Matrix b2;
    Matrix z1;
    Matrix z2;
    Matrix a1;
    Matrix a2;
} ForwardProp;
#endif

ForwardProp* init_parameters(ForwardProp* params, int matrix_size, int target_size);
ForwardProp* forward_propagation(ForwardProp* params, Matrix X, int input_size, int output_size);
void free_forward_prop(ForwardProp* params);
double RELU(double x);
Matrix softmax(Matrix input);
Matrix relu_matrix(Matrix* mat);