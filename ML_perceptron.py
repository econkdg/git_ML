# ====================================================================================================
# perceptron
# ====================================================================================================
# classification

# sum(single perceptron) -> neural network
# ----------------------------------------------------------------------------------------------------
# import library

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------
# generate random data

# linearly separable training data

# num of data
m = 200

# input(x) -> (x1, x2): 2-dimensional
x1 = 8*np.random.rand(m, 1)
x2 = 7*np.random.rand(m, 1) - 4

# classifier(decision boundary)
g = 0.8*x1 + x2 - 3

# output(y) -> color
class_1 = np.where(g >= 1)
class_0 = np.where(g < -1)

class_1 = np.where(g >= 1)[0]
class_0 = np.where(g < -1)[0]

# plot
plt.figure(figsize=(10, 8))
plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
plt.title('linearly separable classes', fontsize=15)
plt.legend(loc=1, fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.show()
# ----------------------------------------------------------------------------------------------------
# perceptron algorithm

# skinny matrix
C1 = np.hstack([np.ones([class_1.shape[0], 1]), x1[class_1], x2[class_1]])
C0 = np.hstack([np.ones([class_0.shape[0], 1]), x1[class_0], x2[class_0]])

X = np.vstack([C1, C0])  # X -> [C1, C0].T
y = np.vstack([np.ones([class_1.shape[0], 1]), -
              np.ones([class_0.shape[0], 1])])

X = np.asmatrix(X)
y = np.asmatrix(y)

# update rule
# (X, y) -> misclassified training point


class perceptron:

    def __init__(self, v1, v2, iter):

        self.v1 = v1
        self.v2 = v2
        self.iter = iter

    def perceptron_algorithm(self):

        w = np.zeros([3, 1])  # random assigned
        w = np.asmatrix(w)

        for k in range(self.iter):

            i_iter = y.shape[0]

            for i in range(i_iter):

                if y[i, 0] != np.sign(X[i, :]*w)[0, 0]:

                    w += y[i, 0]*X[i, :].T

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        predict = np.sign(V[0, :]*w)[0, 0]

        return predict, w

    def perceptron_plot(self):

        x1p = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
        x2p = - self.perceptron_algorithm()[1][1, 0]/self.perceptron_algorithm(
        )[1][2, 0]*x1p - self.perceptron_algorithm()[1][0, 0]/self.perceptron_algorithm()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
        plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
        plt.plot(x1p, x2p, c='k', linewidth=1, label='perceptron')
        plt.xlim([np.min(x1), np.max(x1)])
        plt.xlabel('$x_1$', fontsize=15)
        plt.ylabel('$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=15)

        fig = plt.show()

        return fig


testing_data = [perceptron(4, -2, 150),
                perceptron(4, -2, 10),
                perceptron(4, -2, 1)]
# ----------------------------------------------------------------------------------------------------
# scikit perceptron algorithm


# skinny matrix -> no constant
C1 = np.hstack([x1[class_1], x2[class_1]])
C0 = np.hstack([x1[class_0], x2[class_0]])
X = np.vstack([C1, C0])

y = np.vstack([np.ones([class_1.shape[0], 1]), -
              np.ones([class_0.shape[0], 1])])

# update rule
# (X, y) -> misclassified training point


class perceptron:

    def __init__(self, v1, v2):

        self.v1 = v1
        self.v2 = v2

    def scikit_perceptron(self):

        clf = linear_model.Perceptron(tol=1e-3)

        clf.fit(X, np.ravel(y))

        predict = clf.predict([[self.v1, self.v2]])

        w0 = clf.intercept_[0]
        w1 = clf.coef_[0, 0]
        w2 = clf.coef_[0, 1]

        return predict, w0, w1, w2

    def scikit_perceptron_plot(self):

        x1p = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
        x2p = - self.scikit_perceptron()[2]/self.scikit_perceptron(
        )[3]*x1p - self.scikit_perceptron()[1]/self.scikit_perceptron()[3]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
        plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
        plt.plot(x1p, x2p, c='k', linewidth=1, label='perceptron')
        plt.xlim([np.min(x1), np.max(x1)])
        plt.xlabel('$x_1$', fontsize=15)
        plt.ylabel('$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=15)

        fig = plt.show()

        return fig


testing_data = [perceptron(4, 2),
                perceptron(-2, -1)]
# ====================================================================================================
