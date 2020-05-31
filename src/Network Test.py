import util
import matplotlib.pyplot as plt
from deep_network import deep_network
import numpy as np

def generate_set(n_examples):
    x = np.empty([n_examples,2])
    y = np.empty([n_examples])

    x[:, 0] = np.random.uniform(low=-1, high=1, size=(n_examples,))
    x[:, 1] = np.random.uniform(low=-1, high=1, size=(n_examples,))

    for i in range(n_examples):
        y[i] = np.floor(abs(x[i,0]) + abs(x[i,1]) + np.random.random_sample() * 0.1) # adds some noise to our sample

    return x,y

def plot_points(x, y, save_path):
    """Plot some points where x are the coordinates and y is the label"""
    x_one = x[y > 0.5, :]
    x_two = x[y <= 0.5, :]

    plt.figure()
    plt.scatter(x_one[:, 0], x_one[:, 1], marker='x', color='red')
    plt.scatter(x_two[:, 0], x_two[:, 1], marker='o', color='blue')
    plt.savefig(save_path)

if __name__ == "__main__":
    mind = deep_network([2, 4, 1])
    n = 10000
    alpha = 5e-2
    x, y = generate_set(n)
    loss_change = 1234
    last_loss = 1234
    while np.abs(loss_change) > 1e-4:
        loss = 0
        for i in range(n):
            pred = mind.feed_forward(x[i])
            d_output = pred - y[i]
            loss += 0.5 * np.square(d_output)
            mind.backpropagate(d_output, alpha)
        loss /= n
        loss_change = loss - last_loss
        last_loss = loss
        print(loss)

    n_test = 1000
    x_test, y_test = generate_set(n_test)
    pred_test = np.empty([n_test])
    for i in range(n_test):
        pred_test[i] = (mind.feed_forward(x_test[i]))
    plot_points(x_test,pred_test,"test.png")

