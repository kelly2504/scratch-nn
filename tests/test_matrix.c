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

int main(void) {
    test_transpose();
    test_subtract();
    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}

