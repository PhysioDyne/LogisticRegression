# RegressionImp
Logistic Regression Implementation

This repository contains an implementation of logistic regression in Python using NumPy. Logistic regression is a supervised learning algorithm used for binary classification problems.

## Mathematical Formulas

Logistic regression calculates the probability of an input belonging to a certain class. It uses a logistic (sigmoid) function to map the linear combination of the input features to the range [0, 1].

The sigmoid function is defined as:

sigmoid(x) = 1 / (1 + exp(-x))

The logistic regression algorithm minimizes the following cost function using gradient descent:

J(w, b) = (-1 / n) * sum(y * log(h) + (1 - y) * log(1 - h))

where:
- `J(w, b)` is the cost function
- `w` are the weights
- `b` is the bias
- `n` is the number of samples
- `y` are the true labels
- `h` is the predicted probability

The gradient descent update equations for the weights and bias are:

w = w - alpha * dw

b = b - alpha * db

where:
- `alpha` is the learning rate
- `dw` is the gradient of the weights
- `db` is the gradient of the bias

## References

- [Logistic regression on Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
