#include <stdio.h>
#include "../backward_prop.h"

int main(void) {
    int failures = 0;
    Matrix labels = create_matrix(1, 3);      // labels: 0, 2, 1
    labels.data[0] = 0; labels.data[1] = 2; labels.data[2] = 1;

    Matrix y = one_hot(&labels, 3);           // expect 3 classes x 3 samples
    if (y.rows != 3 || y.cols != 3) { fprintf(stderr, "bad shape\n"); failures++; }
    if (get_matrix_value(&y, 2, 1) != 1.0) { fprintf(stderr, "sample 1 should be class 2\n"); failures++; }
    if (get_matrix_value(&y, 0, 1) != 0.0) { fprintf(stderr, "expected 0\n"); failures++; }

    // derive_relu must return a new matrix and leave its input alone
    Matrix z = create_matrix(1, 3);
    z.data[0] = -2.0; z.data[1] = 0.0; z.data[2] = 3.5;
    Matrix d = derive_relu(&z);
    if (d.data == z.data) { fprintf(stderr, "derive_relu should not reuse input memory\n"); failures++; }
    if (d.data[0] != 0.0 || d.data[1] != 0.0 || d.data[2] != 1.0) { fprintf(stderr, "derive_relu wrong values\n"); failures++; }
    if (z.data[0] != -2.0 || z.data[2] != 3.5) { fprintf(stderr, "derive_relu modified its input\n"); failures++; }

    free(labels.data); free(y.data); free(z.data); free(d.data);
    return failures ? 1 : 0;
}