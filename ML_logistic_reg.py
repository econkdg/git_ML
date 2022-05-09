# ====================================================================================================
# logistic regression
# ====================================================================================================
# import library

import cvxpy as cvx
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------
# sigmoid function

z = np.linspace(-4, 4, 100)
s = 1/(1 + np.exp(-z))

plt.figure(figsize=(10, 2))
plt.plot(z, s)
plt.xlim([-4, 4])
plt.axis('equal')
plt.grid(alpha=0.3)

plt.show()
# ----------------------------------------------------------------------------------------------------
# generate random data

# input(x) -> (x1, x2): 2-dimensional
m = 100

w = np.array([[-6], [2], [1]])

x1 = -25 + 50*np.random.rand(m, 1)
x2 = -25 + 50*np.random.rand(m, 1)

X = np.hstack(
    [np.ones([m, 1]), x1, x2])

w = np.asmatrix(w)
X = np.asmatrix(X)

y = 1/(1 + np.exp(-X*w)) > 0.5

class_1 = np.where(y == True)[0]
class_0 = np.where(y == False)[0]

y = np.empty([m, 1])
y[class_1] = 1
y[class_0] = 0

# plot
plt.figure(figsize=(10, 8))
plt.plot(X[class_1, 1], X[class_1, 2], 'ro', alpha=0.3, label='class_1')
plt.plot(X[class_0, 1], X[class_0, 2], 'bo', alpha=0.3, label='class_0')
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([np.min(X[:, 1]), np.max(X[:, 1])])
plt.ylim([np.min(X[:, 2]), np.max(X[:, 2])])

plt.show()
# ----------------------------------------------------------------------------------------------------
# logistic regression


class logistic:

    def __init__(self, v1, v2, iter):

        self.v1 = v1
        self.v2 = v2
        self.iter = iter

    def logistic_estimation(self):

        def h(x, w):

            return 1/(1 + np.exp(-x*w))

        w = np.zeros([3, 1])
        alpha = 0.01

        for i in range(self.iter):

            df = -X.T*(y - h(X, w))
            w = w - alpha*df

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        if 1/(1 + np.exp(-V[0, :]*w)) > 0.5:
            predict = 1
        else:
            predict = 0

        return predict, w

    def logistic_plot(self):

        xp = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(-1, 1)
        yp = - self.logistic_estimation()[1][1, 0]/self.logistic_estimation(
        )[1][2, 0]*xp - self.logistic_estimation()[1][0, 0]/self.logistic_estimation()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(X[class_1, 1], X[class_1, 2],
                 'ro', alpha=0.3, label='class 1')
        plt.plot(X[class_0, 1], X[class_0, 2],
                 'bo', alpha=0.3, label='class 0')
        plt.plot(xp, yp, 'g', linewidth=4, label='logistic regression')
        plt.title('logistic regression', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(X[:, 1]), np.max(X[:, 1])])
        plt.ylim([np.min(X[:, 2]), np.max(X[:, 2])])

        fig = plt.show()

        return fig


testing_data = [logistic(15, 20, 10000)]
# ----------------------------------------------------------------------------------------------------
# logistic regression(CVXPY)


class logistic_CVXPY:

    def __init__(self, v1, v2):

        self.v1 = v1
        self.v2 = v2

    def logistic_CVXPY_estimation(self):

        w = cvx.Variable([3, 1])

        obj = cvx.Maximize(y.T*X*w - cvx.sum(cvx.logistic(X*w)))
        prob = cvx.Problem(obj).solve()

        w = w.value

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        if 1/(1 + np.exp(-V[0, :]*w)) > 0.5:
            predict = 1
        else:
            predict = 0

        return predict, w

    def logistic_CVXPY_plot(self):

        xp = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(-1, 1)
        yp = - self.logistic_CVXPY_estimation()[1][1, 0]/self.logistic_CVXPY_estimation(
        )[1][2, 0]*xp - self.logistic_CVXPY_estimation()[1][0, 0]/self.logistic_CVXPY_estimation()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(X[class_1, 1], X[class_1, 2],
                 'ro', alpha=0.3, label='class 1')
        plt.plot(X[class_0, 1], X[class_0, 2],
                 'bo', alpha=0.3, label='class 0')
        plt.plot(xp, yp, 'g', linewidth=4, label='logistic regression(CVXPY)')
        plt.title('logistic regression(CVXPY)', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(X[:, 1]), np.max(X[:, 1])])
        plt.ylim([np.min(X[:, 2]), np.max(X[:, 2])])

        fig = plt.show()

        return fig


testing_data = [logistic_CVXPY(15, 20)]
# ----------------------------------------------------------------------------------------------------
# logistic regression(compact)


class logistic_compact:

    def __init__(self, v1, v2):

        self.v1 = v1
        self.v2 = v2

    def logistic_compact_estimation(self):

        y = np.empty([m, 1])
        y[class_1] = 1
        y[class_0] = -1
        y = np.asmatrix(y)

        w = cvx.Variable([3, 1])

        obj = cvx.Minimize(cvx.sum(cvx.logistic(-cvx.multiply(y, X*w))))
        prob = cvx.Problem(obj).solve()

        w = w.value

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        predict = np.sign(V[0, :]*w)[0, 0]

        return predict, w

    def logistic_compact_plot(self):

        xp = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).reshape(-1, 1)
        yp = - self.logistic_compact_estimation()[1][1, 0]/self.logistic_compact_estimation(
        )[1][2, 0]*xp - self.logistic_compact_estimation()[1][0, 0]/self.logistic_compact_estimation()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(X[class_1, 1], X[class_1, 2],
                 'ro', alpha=0.3, label='class 1')
        plt.plot(X[class_0, 1], X[class_0, 2],
                 'bo', alpha=0.3, label='class 0')
        plt.plot(xp, yp, 'g', linewidth=4,
                 label='logistic regression(compact)')
        plt.title('logistic regression(compact)', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(X[:, 1]), np.max(X[:, 1])])
        plt.ylim([np.min(X[:, 2]), np.max(X[:, 2])])

        fig = plt.show()

        return fig


testing_data = [logistic_compact(15, 20)]
# ----------------------------------------------------------------------------------------------------
# logistic regression(scikit learn)


class logistic_scikit:

    def __init__(self, v1, v2):

        self.v1 = v1
        self.v2 = v2

    def logistic_scikit_estimation(self):

        X = np.hstack([x1, x2])

        clf = linear_model.LogisticRegression(solver='lbfgs')
        clf.fit(X, np.ravel(y))

        w0 = clf.intercept_[0]
        w1 = clf.coef_[0, 0]
        w2 = clf.coef_[0, 1]

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        if 1/(1 + np.exp(-V[0, :]*w)) > 0.5:
            predict = 1
        else:
            predict = 0

        return predict, w0, w1, w2

    def logistic_scikit_plot(self):

        xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
        yp = - self.logistic_scikit_estimation()[2]/self.logistic_scikit_estimation(
        )[3]*xp - self.logistic_scikit_estimation()[1]/self.logistic_scikit_estimation()[3]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(X[class_1, 1], X[class_1, 2],
                 'ro', alpha=0.3, label='class 1')
        plt.plot(X[class_0, 1], X[class_0, 2],
                 'bo', alpha=0.3, label='class 0')
        plt.plot(xp, yp, 'g', linewidth=4,
                 label='logistic regression(scikit)')
        plt.title('logistic regression(scikit)', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(X[:, 1]), np.max(X[:, 1])])
        plt.ylim([np.min(X[:, 2]), np.max(X[:, 2])])

        fig = plt.show()

        return fig


testing_data = [logistic_scikit(15, 20)]
# ====================================================================================================
