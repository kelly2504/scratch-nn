#include "metrics.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Keeps log() finite if softmax outputs exactly 0
#define LOG_EPSILON 1e-12

static void check_shapes(const char* fn, const Matrix* a2, const Matrix* labels) {
    int m = labels->rows * labels->cols;
    if (a2->cols != m) {
        fprintf(stderr, "%s: a2 has %d columns but there are %d labels\n", fn, a2->cols, m);
        exit(EXIT_FAILURE);
    }
}

Matrix get_predictions(const Matrix* a2) {
    Matrix preds = create_matrix(1, a2->cols);

    // for each sample (column), find the class (row) with the highest probability
    for (int j = 0; j < a2->cols; j++) {
        int best = 0;
        double best_val = a2->data[j];  // row 0, column j
        for (int i = 1; i < a2->rows; i++) {
            double v = a2->data[i * a2->cols + j];
            if (v > best_val) {
                best_val = v;
                best = i;
            }
        }
        preds.data[j] = best;
    }

    return preds;
}

double get_accuracy(const Matrix* a2, const Matrix* labels) {
    check_shapes("get_accuracy", a2, labels);

    Matrix preds = get_predictions(a2);
    int m = preds.cols;
    int correct = 0;

    for (int j = 0; j < m; j++) {
        if ((int)preds.data[j] == (int)labels->data[j]) {
            correct++;
        }
    }

    free(preds.data);
    return (double)correct / m;
}

double cross_entropy_loss(const Matrix* a2, const Matrix* labels) {
    check_shapes("cross_entropy_loss", a2, labels);

    int m = a2->cols;
    double total = 0.0;

    // one_hot_Y is 1 only at the true class, so only that row's log contributes
    for (int j = 0; j < m; j++) {
        int label = (int)labels->data[j];
        if (label < 0 || label >= a2->rows) {
            fprintf(stderr, "cross_entropy_loss: label %d at index %d out of range\n", label, j);
            exit(EXIT_FAILURE);
        }
        double p = a2->data[label * a2->cols + j];
        total += log(p + LOG_EPSILON);
    }

    return -total / m;
}
