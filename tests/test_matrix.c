#include <stdio.h>
#include <math.h>
#include "../matrix.h"

static int failures = 0;
#define CHECK(cond) do {if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++;} } while (0)
#define CHECK_NEAR(a, b) CHECK(fabs((a) - (b)) < 1e-9) 

static void test_transpose(void) {
    Matrix m = create_matrix(2, 3);
    for (int i = 0; i < 6; i++) m.data[i] = i + 1;

    Matrix t = matrix_transpose(m);
    CHECK(t.rows == 3 && t.cols == 2);
    CHECK_NEAR(get_matrix_value(&t, 0, 1), 4.0);
    CHECK_NEAR(get_matrix_value(&t, 2, 0), 3.0);
}

static void test_subtract(void) {
    Matrix a = create_matrix(1, 2), b = create_matrix(1, 2);
    a.data[0] = 5; a.data[1] = 3;
    b.data[0] = 2; b.data[1] = 1;
    Matrix c = subtract_matrices(a, b);
    CHECK_NEAR(c.data[0], 3.0);
    CHECK_NEAR(c.data[1], 2.0);
}

static void test_elementwise_multiply(void) {
    Matrix a = create_matrix(2, 2), b = create_matrix(2, 2);
    double av[] = {1, 2, 3, 4}, bv[] = {5, 6, 0, -1};
    for (int i = 0; i < 4; i++) { a.data[i] = av[i]; b.data[i] = bv[i]; }

    Matrix c = matrix_elementwise_multiply(&a, &b);
    CHECK(c.rows == 2 && c.cols == 2);
    CHECK_NEAR(c.data[0], 5.0);
    CHECK_NEAR(c.data[1], 12.0);
    CHECK_NEAR(c.data[2], 0.0);
    CHECK_NEAR(c.data[3], -4.0);
    CHECK_NEAR(a.data[0], 1.0);   // inputs unchanged
    free(a.data); free(b.data); free(c.data);
}

int main(void) {
    test_transpose();
    test_subtract();
    test_elementwise_multiply();
    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}

