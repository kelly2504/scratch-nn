#ifndef METRICS_H
#define METRICS_H

#include "matrix.h"

// a2:     softmax output, shape (num_classes x m), one column per sample
// labels: true class for each sample, shape (1 x m) or (m x 1)

// Index of the highest probability in each column of a2 -> (1 x m)
// Like np.argmax(A2, 0)
Matrix get_predictions(const Matrix* a2);

// Fraction of samples where the prediction equals the label, in [0, 1]
double get_accuracy(const Matrix* a2, const Matrix* labels);

// Mean cross-entropy loss: -(1/m) * sum(log(a2[label_i, i]))
// Same as -(1/m) * np.sum(one_hot_Y * np.log(A2))
double cross_entropy_loss(const Matrix* a2, const Matrix* labels);

#endif
