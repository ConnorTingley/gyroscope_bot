import numpy as np
class deep_network:
    a = 0
    W = 0
    b = 0
    W_update = 0
    b_update = 0
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
        self.W_update = []
        for s in range(len(shape)-1):
            self.W.append(np.random.rand(shape[s+1],shape[s]))
            self.W_update.append(np.empty([shape[s + 1], shape[s]]))
        self.b = []
        self.b_update = []
        self.z = []
        self.d = []
        for s in range(1, len(shape)):
            self.b.append(np.random.rand(shape[s]))
            self.b_update.append(np.empty([shape[s]]))
            self.z.append(np.empty([shape[s]]))
            self.d.append(np.empty([shape[s]]))
        self.n_layers = len(shape) - 1

    def feed_forward(self, input):
        self.a[0] = input
        for i in range(0, self.n_layers):
            self.z[i] = self.W[i] @ self.a[i] + self.b[i]
            self.a[i+1] = sigmoid(self.z[i])
        return self.a[self.n_layers]

    def backpropagate_no_update(self, d_output, learning_rate):
        for i in range(self.n_layers - 1, -1, -1):
            if i == self.n_layers - 1:
                self.d[i] = d_output
            else:
                self.d[i] = np.multiply(self.W[i + 1].T @ self.d[i+1], dsigmoid(self.z[i]))

    def backpropagate(self, d_output, learning_rate):
        for i in range(self.n_layers - 1, -1, -1):
            if i == self.n_layers - 1:
                self.d[i] = d_output
            else:
                self.d[i] = np.multiply(self.W[i + 1].T @ self.d[i+1], dsigmoid(self.z[i]))
            # Update
            self.W_update[i] += np.outer(self.d[i], self.a[i]) * learning_rate
            self.b_update[i] += self.d[i] * learning_rate

    def update(self):
        for i in range(self.n_layers - 1, -1, -1):
            self.W[i] = self.W[i] - self.W_update[i]
            self.b[i] = self.b[i] - self.b_update[i]

        for array in self.W_update:
            array.fill(0)
        for array in self.b_update:
            array.fill(0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    sig = sigmoid(z)
    return np.multiply(sig,(1-sig))
