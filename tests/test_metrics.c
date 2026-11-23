#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../metrics.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)
#define CHECK_NEAR(a, b) CHECK(fabs((a) - (b)) < 1e-9)

// 3 classes x 4 samples, each column sums to 1
//            s0   s1   s2   s3
// class 0   0.7  0.1  0.2  0.3
// class 1   0.2  0.8  0.3  0.3
// class 2   0.1  0.1  0.5  0.4
static Matrix make_a2(void) {
    double v[] = {0.7, 0.1, 0.2, 0.3,
                  0.2, 0.8, 0.3, 0.3,
                  0.1, 0.1, 0.5, 0.4};
    Matrix a2 = create_matrix(3, 4);
    for (int i = 0; i < 12; i++) a2.data[i] = v[i];
    return a2;
}

static Matrix make_labels(double l0, double l1, double l2, double l3) {
    Matrix y = create_matrix(1, 4);
    y.data[0] = l0; y.data[1] = l1; y.data[2] = l2; y.data[3] = l3;
    return y;
}

static void test_predictions(void) {
    Matrix a2 = make_a2();
    Matrix p = get_predictions(&a2);
    CHECK(p.rows == 1 && p.cols == 4);
    CHECK_NEAR(p.data[0], 0);
    CHECK_NEAR(p.data[1], 1);
    CHECK_NEAR(p.data[2], 2);
    CHECK_NEAR(p.data[3], 2);
    free(a2.data); free(p.data);
}

static void test_accuracy(void) {
    Matrix a2 = make_a2();
    Matrix all_right = make_labels(0, 1, 2, 2);
    Matrix half_right = make_labels(0, 1, 0, 0);
    CHECK_NEAR(get_accuracy(&a2, &all_right), 1.0);
    CHECK_NEAR(get_accuracy(&a2, &half_right), 0.5);
    free(a2.data); free(all_right.data); free(half_right.data);
}

static void test_loss(void) {
    Matrix a2 = make_a2();
    Matrix y = make_labels(0, 1, 2, 0);
    // -(log 0.7 + log 0.8 + log 0.5 + log 0.3) / 4
    double expected = -(log(0.7) + log(0.8) + log(0.5) + log(0.3)) / 4.0;
    CHECK(fabs(cross_entropy_loss(&a2, &y) - expected) < 1e-9);

    // a perfect prediction should give (almost) zero loss
    Matrix perfect = create_matrix(2, 1);
    perfect.data[0] = 1.0; perfect.data[1] = 0.0;
    Matrix y0 = create_matrix(1, 1);
    CHECK(cross_entropy_loss(&perfect, &y0) < 1e-9);

    free(a2.data); free(y.data); free(perfect.data); free(y0.data);
}

int main(void) {
    test_predictions();
    test_accuracy();
    test_loss();
    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}
