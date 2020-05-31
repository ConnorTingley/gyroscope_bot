import numpy as np
class network:
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
        self.a = np.empty(shape)
        self.w = []
        for s in shape:
            self.w.append(np.empty[s,s])
        self.b = np.empty(shape)
        self.z = np.empty(shape)
        self.d = np.empty(shape)
        self.n_layers = len(shape)

    def feed_forward(self, input):
        self.a[0] = input
        for i in range(0, self.n_layers):
            self.z[i] = self.W[i] @ self.a[i] + self.b[i]
            self.a[i+1] = sigmoid(self.z[i])

    def backpropagate(self, d_output, learning_rate):
        self.d[self.n_layers - 1] = d_output
        for i in range(self.n_layers - 2, -1):
            self.d[i] = np.multiply(self.W[i+1] @ self.d[i+1], dsigmoid(self.z[i]))
            # Update
            self.W[i] = self.W[i] - np.outer(self.d[i], self.a[i]) * learning_rate
            self.b[i] = self.b[i] - self.d[i] * learning_rate

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    sig = sigmoid(z)
    return np.multiply(sig,(1-sig))
