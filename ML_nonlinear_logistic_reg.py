# ====================================================================================================
# nonlinear logistic regression
# ====================================================================================================
# import library

import cvxpy as cvx
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------
# generate random data

# non-linearly separable training data

# input(x) -> (x1, x2): 2-dimensional
class_1 = np.array([[-1.1, 0], [-0.3, 0.1], [-0.9, 1], [0.8, 0.4], [0.4, 0.9], [0.3, -0.6],
                    [-0.5, 0.3], [-0.8, 0.6], [-0.5, -0.5]])

class_0 = np.array([[-1, -1.3], [-1.6, 2.2], [0.9, -0.7], [1.6, 0.5], [1.8, -1.1], [1.6, 1.6],
                    [-1.6, -1.7], [-1.4, 1.8], [1.6, -0.9], [0, -1.6], [0.3, 1.7], [-1.6, 0], [-2.1, 0.2]])

class_1 = np.asmatrix(class_1)
class_0 = np.asmatrix(class_0)

data = np.vstack([class_1, class_0])

# plot
plt.figure(figsize=(10, 8))
plt.plot(class_1[:, 0], class_1[:, 1], 'ro', label='class 1')
plt.plot(class_0[:, 0], class_0[:, 1], 'bo', label='class 0')
plt.title('logistic regression for nonlinear data', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')

plt.show()
# ----------------------------------------------------------------------------------------------------
# kernel method


class nonlinear_logistic:

    def __init__(self, gamma):

        self.gamma = gamma

    def logistic_compact_form(self):

        N = class_1.shape[0]
        M = class_0.shape[0]

        X = np.vstack([class_1, class_0])
        y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])

        X = np.asmatrix(X)
        y = np.asmatrix(y)

        m = N + M
        Z = np.hstack([np.ones([m, 1]), np.sqrt(2)*X[:, 0], np.sqrt(2)*X[:, 1], np.square(
            X[:, 0]), np.sqrt(2)*np.multiply(X[:, 0], X[:, 1]), np.square(X[:, 1])])

        # compact form
        w = cvx.Variable([6, 1])
        obj = cvx.Minimize(cvx.sum(cvx.logistic(-cvx.multiply(y, Z*w))))
        prob = cvx.Problem(obj).solve()

        w = w.value

        return w

    def logistic_compact_form_plot(self):

        [X1gr, X2gr] = np.meshgrid(
            np.arange(np.min(data[:, 0]), np.max(data[:, 0]), 0.1), np.arange(np.min(data[:, 1]), np.max(data[:, 1]), 0.1))

        Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
        Xp = np.asmatrix(Xp)

        m = Xp.shape[0]

        Zp = np.hstack([np.ones([m, 1]), np.sqrt(2)*Xp[:, 0], np.sqrt(2)*Xp[:, 1], np.square(Xp[:, 0]),
                        np.sqrt(2)*np.multiply(Xp[:, 0], Xp[:, 1]), np.square(Xp[:, 1])])
        q = Zp*self.logistic_compact_form()

        B = []

        for i in range(m):

            if q[i, 0] > 0:

                B.append(Xp[i, :])

        B = np.vstack(B)

        plt.figure(figsize=(10, 8))
        plt.plot(class_1[:, 0], class_1[:, 1], 'ro', label='class 1')
        plt.plot(class_0[:, 0], class_0[:, 1], 'bo', label='class 0')
        plt.plot(B[:, 0], B[:, 1], 'gs', markersize=10,
                 alpha=0.1, label='logistic regression')
        plt.title('logistic regression with kernel', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
        plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])

        fig = plt.show()

        return fig


testing_data = [nonlinear_logistic(2),
                nonlinear_logistic(3),
                nonlinear_logistic(4)]
# ====================================================================================================
