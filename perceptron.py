import numpy as np


class Perceptron:
    def __init__(self, w, b, lr):
        self.w = w
        self.b = b
        self.lr = lr

    def _is_right(self, x, y):
        return True if y * (np.dot(x, np.reshape(self.w, [2, 1])).sum() + self.b) > 0 else False

    def train(self, data):
        for sample in data:
            x = sample[:, :2]
            y = sample[0, -1]
            # print(np.shape(x), y)

            if not self._is_right(x, y):
                # print(np.shape(x), np.shape(self.w))
                self.w += self.lr * y * x
                self.b += self.lr * y
                self.print_arg()
                self.train(data)
                return 0
        return 0

    def predict(self, x):
        return 1 if np.dot(x, np.reshape(self.w, [2, 1])).sum() + self.b > 0 else -1

    def print_arg(self):
        print(self.w, self.b)
