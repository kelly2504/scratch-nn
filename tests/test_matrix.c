#include <stdio.h>
#include <math.h>
#include <time.h>
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

static void test_gemm_speed() {
    /*
    naive (as of 1774309dca7066cba7f894295ad2d6c82a2360bf) = 1.309098
    transpose first (as of 949d9bb37550cdf35ebec24d878a491a95f60047) = 1.121513
    
    */

    int num_tests = 20;
    int mat_size = 500;
    double times[num_tests];
    int it = 0;

    Matrix mat1 = create_matrix(mat_size, mat_size);
    for (int x = 0; x < mat1.cols; x++) {
        for (int y = 0; y < mat1.rows; y++) {
            set_matrix_value(&mat1, x, y, 1.0);
        }
    }

    Matrix mat2 = create_matrix(mat_size, mat_size);
    for (int x = 0; x < mat2.cols; x++) {
        for (int y = 0; y < mat2.rows; y++) {
            set_matrix_value(&mat2, x, y, 1.0);
        }
    }


    for (int i = 0; i < num_tests; i++) {
        clock_t start_time =  clock();

        Matrix ret = matrix_multiply(&mat1, &mat2);

        times[it++] = (clock() - start_time) / (float) CLOCKS_PER_SEC;

        free(ret.data);
    }

    free(mat1.data);
    free(mat2.data);

    double s = 0;
    for (int i = 0; i < num_tests; i++) {
        s += times[i];
    }

    printf("av mult time = %f\n", s / num_tests);
}

int main(void) {
    test_transpose();
    test_subtract();
    test_elementwise_multiply();
    test_gemm_speed();
    if (failures) fprintf(stderr, "%d check(s) failed\n", failures);
    return failures ? 1 : 0;
}

