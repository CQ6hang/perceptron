import numpy as np

from perceptron import Perceptron


def run_this(data, test, w, b, lr):
    perceptron = Perceptron(w, b, lr)

    perceptron.print_arg()

    perceptron.train(data)

    print(perceptron.predict(test))


if __name__ == '__main__':
    data = [np.array([[3, 3, 1]]), np.array([[4, 3, 1]]), np.array([[1, 1, -1]])]
    test = np.array([[2, 5]])

    # print(np.shape(data), np.shape(test))

    # print(np.reshape(np.array([[2, 5]]), [2, 1]))

    # print(np.dot(np.array([[2, 2]]),np.array([[1], [3]])).sum())

    run_this(data, test, np.array([[0, 0]]), 0, 1)
