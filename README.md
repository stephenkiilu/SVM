# SVM from Scratch

This repository demonstrates how to build a minimal linear Support Vector Machine (SVM) classifier using only the Python standard library. It is intended for educational purposes and shows how the core learning algorithm works without relying on scientific computing packages such as NumPy or scikit-learn.

## Features

- Plain Python implementation with no external dependencies
- Stochastic gradient descent optimization with hinge loss
- Easy to read and extend for experiments
- Includes a small demo on an OR-gate dataset

## File Overview

- `svm.py` – Contains the `LinearSVM` class and a `_demo` function that trains the model on example data.

## Requirements

Any Python 3 interpreter should work. The code does not depend on third‑party libraries, so no additional installation steps are required.

## Running the Example

Execute the script directly to train a simple model and view the predictions:

```bash
python3 svm.py
```

You should see output similar to the following:

```
Predictions: [ -1, 1, 1, 1 ]
```

This output corresponds to the OR-gate classification task used in the demo.

## Using the `LinearSVM` Class

The classifier exposes two main methods:

- `fit(X, y)` – Train the model on a list of feature vectors `X` and corresponding labels `y` (either `1` or `-1`).
- `predict(X)` – Return predictions for a list of feature vectors.

You can import the class in your own scripts and supply any linearly separable dataset.

## Extending the Code

The implementation is deliberately compact so that you can easily experiment with the underlying algorithm. Possible extensions include:

- Implementing kernel methods for non-linear classification
- Adding support for saving/loading trained models
- Comparing performance with established libraries

## License

This project is released into the public domain.

