import numpy as np
class deep_network:

    def __init__(self, shape, activation, d_activation):
        self.shape = shape
        self.activation = activation
        self.d_activation = d_activation
        self.W = []
        for s in range(len(shape)-1):
            self.W.append(np.random.uniform(low =-1,high = 1,size =(shape[s],shape[s+1])))
        self.b = []
        self.d = []
        for s in range(1, len(shape)):
            self.b.append(np.zeros(shape[s]))
            self.d.append(np.zeros(shape[s]))
        self.n_layers = len(shape) - 1

    def feed_forward(self, input):
        a = [input]
        z = []
        for i in range(0, self.n_layers):
            z.append(np.add((a[i] @ self.W[i]) / self.shape[i], self.b[i]))
            a.append(self.activation(z[i]) if i != self.n_layers-1 else z[i])
        return a[-1], a, z

    def backpropagate(self, d_output, a, z, learning_rate, reg):
        grad_W = [0] * self.n_layers
        grad_b = [0] * self.n_layers
        n_examples = a[0].shape[0]
        for i in range(self.n_layers - 1, -1, -1):
            self.d[i] = d_output if i == self.n_layers - 1 else np.multiply(self.d[i+1] @ self.W[i + 1].T, self.d_activation(z[i]))
            # Grad
            grad_W[i] = a[i].T @ self.d[i] / n_examples + 2 * reg * self.W[i]
            grad_b[i] = np.mean(self.d[i], axis=0)
            self.W[i] -= grad_W[i] * (learning_rate)
            self.b[i] -= grad_b[i] * (learning_rate)
            #print("W:",self.W[i])
            #print("B:", self.b[i])
            #print("GW:",grad_W[i])
            #print("GB:", grad_b[i])
            #print("d:", np.mean(np.abs(self.d[i]), axis=0))
        #print("---")
        return self.d[0]  @ self.W[0].T
