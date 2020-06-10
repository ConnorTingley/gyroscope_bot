import util
import matplotlib.pyplot as plt
from deep_network import deep_network
import numpy as np

def generate_set(n_examples):
    x = np.empty([n_examples,2])
    y = np.empty([n_examples,1])

    x[:, 0] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    x[:, 1] = np.random.uniform(low=-1, high=1, size=(n_examples,))

    for i in range(n_examples):
        y[i,0] = np.square(1*(x[i,0] + 0.25)) + np.square(1*(x[i,1] + 0.25)) + np.random.random_sample() * 0.1 # adds some noise to our sample

    return x,y

def plot_points(x, y, save_path):
    """Plot some points where x are the coordinates and y is the label"""
    x_one = x[y[:,0] > 0.5]
    x_two = x[y[:,0] <= 0.5]

    plt.figure()
    plt.scatter(x_one[:, 0], x_one[:, 1], marker='x', color='red')
    plt.scatter(x_two[:, 0], x_two[:, 1], marker='o', color='blue')
    plt.savefig(save_path)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(z):
    sig = sigmoid(z)
    return np.multiply(sig,(1-sig))

if __name__ == "__main__":
    mind = deep_network([2, 4, 1], sigmoid, dsigmoid)
    #print(mind.W)
    #print(mind.b)
    n = 10000
    alpha = 0.1
    n_b = 50
    x, y = generate_set(n)
    split_x = np.split(x, n / n_b)
    split_y = np.split(y, n / n_b)

    for e in range(200):
        for i in range(len(split_x)):
            pred, a, z = mind.feed_forward(split_x[i])
            d_output = (pred - split_y[i])
            accuracy = 1- np.mean(np.abs(d_output))
            mind.backpropagate(d_output, a, z, alpha, 1e-6)
        print("Acc:", accuracy)

    #print(mind.W)
    #print(mind.b)

    n_test = 1000
    x_test, y_test = generate_set(n_test)
    pred_test, a, z = (mind.feed_forward(x_test))
    plot_points(x_test, pred_test, "test.png")