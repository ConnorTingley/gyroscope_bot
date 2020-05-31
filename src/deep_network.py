import numpy as np
class deep_network:
    a = 0
    W = 0
    b = 0
    z = 0
    d = 0
    d_batch_sum = 0
    batch_number = 0
    shape = 0
    n_layers = 0

    def __init__(self, shape):
        self.shape = shape
        self.a = []
        for s in range(len(shape)):
            self.a.append(np.random.rand(shape[s]))
        self.W = []
        for s in range(len(shape)-1):
            self.W.append(np.random.rand(shape[s+1],shape[s]))
        self.b = []
        self.z = []
        self.d = []
        for s in range(1, len(shape)):
            self.b.append(np.random.rand(shape[s]))
            self.z.append(np.empty([shape[s]]))
            self.d.append(np.empty([shape[s]]))
        self.n_layers = len(shape) - 1

    def feed_forward(self, input):
        self.a[0] = input
        for i in range(0, self.n_layers):
            self.z[i] = self.W[i] @ self.a[i] + self.b[i]
            self.a[i+1] = sigmoid(self.z[i])
        return self.a[self.n_layers]

    def backpropagate(self, d_output, learning_rate):
        for i in range(self.n_layers - 1, -1, -1):
            if i == self.n_layers - 1:
                self.d[i] = d_output
            else:
                self.d[i] = np.multiply(self.W[i + 1].T @ self.d[i+1], dsigmoid(self.z[i]))
            # Update
            self.W[i] = self.W[i] - np.outer(self.d[i], self.a[i]) * learning_rate
            self.b[i] = self.b[i] - self.d[i] * learning_rate


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    sig = sigmoid(z)
    return np.multiply(sig,(1-sig))
