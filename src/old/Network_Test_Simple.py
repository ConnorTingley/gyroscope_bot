import util
import matplotlib.pyplot as plt
from deep_network import deep_network
import numpy as np

def generate_set(n_examples):
    x = np.empty([n_examples,6])
    y = np.empty([n_examples,1])

    x[:, 0] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    x[:, 1] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    x[:, 2] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    x[:, 3] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    x[:, 4] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    x[:, 5] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    y[:, 0] = np.sum(x, axis=1)

    return x,y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    sig = sigmoid(z)
    return np.multiply(sig,(1-sig))

def linear(z):
    return z

def d_linear(z):
    return 1

if __name__ == "__main__":
    mind = deep_network([6,1], linear, d_linear)
    #print(mind.W)
    #print(mind.b)
    n = 1000
    alpha = 1
    x, y = generate_set(n)
    for e in range(20):
        pred, a, z = mind.feed_forward(x)
        d_output = (pred - y)
        accuracy = 1- np.mean(np.abs(d_output))
        mind.backpropagate(d_output, a, z, alpha,0.001)
        print("Acc:", accuracy)

    #print(mind.W)
    #print(mind.b)

    n_test = 1000
    x_test, y_test = generate_set(n_test)
    pred_test, a, z = (mind.feed_forward(x_test))
    plot_points(x_test,pred_test,"test.png")