#ifndef BACKWARD_PROP_H
#define BACKWARD_PROP_H

#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "forward_prop.h"

typedef struct BackwardProp {
    Matrix dw1;
    Matrix dw2;
    Matrix db1;
    Matrix db2;
    Matrix dz1;
    Matrix dz2;
}BackwardProp;
#endif

Matrix one_hot(const Matrix* labels, int num_classes);
BackwardProp backward_propagation(ForwardProp* params, Matrix X, Matrix Y);
Matrix derive_relu(const Matrix* mat);
ForwardProp* update_params(ForwardProp* params, BackwardProp* derivatives, double alpha);
void free_backward_prop(BackwardProp* backprop);