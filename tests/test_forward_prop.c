#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../forward_prop.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

static void test_softmax_columns_sum_to_one(void) {
    Matrix z = create_matrix(3, 2);
    double v[] = {1.0, 1000.0,
                  2.0, 1001.0,    // large values: must not overflow
                  3.0, 1002.0};
    for (int i = 0; i < 6; i++) z.data[i] = v[i];

    Matrix s = softmax(z);
    CHECK(s.rows == 3 && s.cols == 2);
    for (int j = 0; j < 2; j++) {
        double sum = 0;
        for (int i = 0; i < 3; i++) sum += s.data[i * 2 + j];
        CHECK(fabs(sum - 1.0) < 1e-9);
    }
    // largest input gets largest probability; both columns have the same gaps
    CHECK(s.data[4] > s.data[2] && s.data[2] > s.data[0]);
    CHECK(fabs(s.data[0] - s.data[1]) < 1e-9);
    free(z.data); free(s.data);
}

static void test_relu_does_not_modify_input(void) {
    Matrix z = create_matrix(1, 2);
    z.data[0] = -1.0; z.data[1] = 2.0;
    Matrix a = relu_matrix(&z);
    CHECK(a.data[0] == 0.0 && a.data[1] == 2.0);
    CHECK(z.data[0] == -1.0);
    CHECK(a.data != z.data);
    free(z.data); free(a.data);
}

static void test_forward_shapes(void) {
    int features = 6, classes = 4, m = 5;
    ForwardProp p;
    init_parameters(&p, features, classes);

    Matrix X = create_matrix(features, m);
    for (int i = 0; i < features * m; i++) X.data[i] = (i % 7) / 7.0;

    forward_propagation(&p, X, features, classes);
    CHECK(p.z1.rows == classes && p.z1.cols == m);
    CHECK(p.a1.rows == classes && p.a1.cols == m);
    CHECK(p.z2.rows == classes && p.z2.cols == m);
    CHECK(p.a2.rows == classes && p.a2.cols == m);

    for (int j = 0; j < m; j++) {
        double sum = 0;
        for (int i = 0; i < classes; i++) sum += p.a2.data[i * m + j];
        CHECK(fabs(sum - 1.0) < 1e-9);
    }
}

int main(void) {
    test_softmax_columns_sum_to_one();
    test_relu_does_not_modify_input();
    test_forward_shapes();
    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
