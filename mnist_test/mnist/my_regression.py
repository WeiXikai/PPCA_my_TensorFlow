import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist_data = read_data_sets('mnist_data', one_hot = True)

class LogisticRegression:

    def __init__(self):
        # W: matrix of 784 * 10
        # x: input image, matrix of 1 * 784
        # b: matrix of 1 * 10
        self.n = 784
        self.m = 10
        self.W = np.random.normal(size = (784, 10))
        self.W = self.W / np.sum(self.W)
        self.b = np.zeros([1, 10])
        self.step = 0.01

    def softmax(self, x):
        return np.exp(np.dot(x, self.W) + self.b) / np.sum(np.exp(np.dot(x, self.W) + self.b))

    def grad_loss(self, batch):
        # use cross_entropy
        # dC/db_j = sfmx_j - y_j
        # dC/dW_jk = x_k(sfmx_j - y_j)
        # pb = sfmx - y
        # pW = np.dot(x.T, sfmx - y)
        num = len(batch[0])
        dW = 0
        db = 0
        for i in range(1, num):
            x = np.reshape(batch[0][i], (784, 1))
            x = x.T
            y = np.reshape(batch[1][i], (10, 1))
            y = y.T
            sfmx = self.softmax(x)
            dW += np.dot(x.T, sfmx - y)
            db += sfmx - y
        self.W = self.W - self.step * dW
        self.b = self.b - self.step * db
        self.step *= 0.999


    def predict_accuarcy(self, test_batch):
        total = len(test_batch[0])
        right = 0
        for i in range(1, total):
            x = np.reshape(test_batch[0][i], (784, 1))
            x = x.T
            y = np.reshape(test_batch[1][i], (10, 1))
            y = y.T
            sfmx = self.softmax(x)
            right += np.equal(np.argmax(sfmx, 1), np.argmax(y, 1))
        return right / total

if __name__ == '__main__':
    LG = LogisticRegression()

    for i in range(2000):
        batch = mnist_data.train.next_batch(100)
        LG.grad_loss(batch)
        if i % 100 == 0:
            test_batch = mnist_data.test.next_batch(100)
            print("step %d, training accuracy %g" % (i, LG.predict_accuarcy(test_batch)))
