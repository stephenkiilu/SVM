"""Simple linear SVM from scratch using plain Python.

This module implements a basic linear Support Vector Machine (SVM)
using stochastic gradient descent. It does not rely on external
libraries such as NumPy.
"""

from __future__ import annotations

import random
from typing import List, Tuple


class LinearSVM:
    """Linear SVM classifier using hinge loss."""

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights: List[float] | None = None
        self.bias: float = 0.0

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        """Train the classifier.

        Parameters
        ----------
        X : List of feature vectors.
        y : List of labels (1 or -1).
        """
        if not X:
            raise ValueError("Training data X cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        n_features = len(X[0])
        # Initialize weights with zeros
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0

        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                condition = yi * (self._dot(xi, self.weights) - self.bias) >= 1
                if condition:
                    # Only apply regularization
                    self.weights = [w - self.learning_rate * (2 * self.lambda_param * w) for w in self.weights]
                else:
                    # Update weights and bias with hinge loss gradient
                    self.weights = [
                        w - self.learning_rate * (2 * self.lambda_param * w - yi * xij)
                        for w, xij in zip(self.weights, xi)
                    ]
                    self.bias -= self.learning_rate * yi

    def predict(self, X: List[List[float]]) -> List[int]:
        if self.weights is None:
            raise ValueError("Model is not trained yet")
        predictions = []
        for xi in X:
            linear_output = self._dot(xi, self.weights) - self.bias
            predictions.append(1 if linear_output >= 0 else -1)
        return predictions

    @staticmethod
    def _dot(x: List[float], w: List[float]) -> float:
        return sum(x_i * w_i for x_i, w_i in zip(x, w))


def _demo() -> None:
    """Run a simple demo on linearly separable data."""
    # Toy dataset: OR gate
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    # Labels: -1 for negative class, 1 for positive class
    y = [-1, 1, 1, 1]

    # Shuffle data for training
    combined = list(zip(X, y))
    random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)

    svm = LinearSVM(learning_rate=0.1, lambda_param=0.01, epochs=100)
    svm.fit(list(X_shuffled), list(y_shuffled))

    predictions = svm.predict(X)
    print("Predictions:", predictions)


if __name__ == "__main__":
    _demo()
