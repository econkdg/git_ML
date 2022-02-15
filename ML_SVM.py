# ====================================================================================================
# SVM(support vector machine)
# ====================================================================================================
# import library

import cvxpy as cvx
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------
# generate random data

# linearly separable training data

# num of data
m = 100

# input(x) -> (x1, x2): 2-dimensional
x1 = 8*np.random.rand(m, 1)
x2 = 7*np.random.rand(m, 1) - 4

# classifier(decision boundary)
g = 0.8*x1 + x2 - 3
g1 = g - 1
g0 = g + 1

# output(y) -> color
class_1 = np.where(g1 >= 0)[0]
class_0 = np.where(g0 < 0)[0]

xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
ypt = -0.8*xp + 3

# plot 1
plt.figure(figsize=(10, 8))
plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
plt.plot(xp, ypt, 'k', linewidth=3, label='True')
plt.title('linearly and strictly separable classes', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([np.min(x1), np.max(x1)])
plt.ylim([np.min(x2), np.max(x2)])

plt.show()

# plot 2
xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
ypt = -0.8*xp + 3

plt.figure(figsize=(10, 8))
plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
plt.plot(xp, ypt, 'k', linewidth=3, label='True')
plt.plot(xp, ypt-1, '--k')
plt.plot(xp, ypt+1, '--k')
plt.title('linearly and strictly separable classes', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([np.min(x1), np.max(x1)])
plt.ylim([np.min(x2), np.max(x2)])

plt.show()
# ----------------------------------------------------------------------------------------------------
# CVXPY_1: 1st attempt


class CVXPY_1:

    def __init__(self, v1, v2):

        self.v1 = v1
        self.v2 = v2

    def CVXPY_1_form_1(self):

        X1 = np.hstack([x1[class_1], x2[class_1]])
        X0 = np.hstack([x1[class_0], x2[class_0]])

        X1 = np.asmatrix(X1)
        X0 = np.asmatrix(X0)

        N = X1.shape[0]
        M = X0.shape[0]

        w0 = cvx.Variable([1, 1])
        w = cvx.Variable([2, 1])

        obj = cvx.Minimize(1)
        const = [w0 + X1*w >= 1, w0 + X0*w <= -1]
        prob = cvx.Problem(obj, const).solve()

        w0 = w0.value
        w = w.value

        V = [self.v1, self.v2]
        V = np.asmatrix(V)

        predict = np.sign(w0 + V[0, :]*w)[0, 0]

        return predict, w0, w

    def CVXPY_1_form_1_plot(self):

        xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
        yp = - self.CVXPY_1_form_1()[2][0, 0]/self.CVXPY_1_form_1()[2][1, 0] * \
            xp - self.CVXPY_1_form_1()[1][0]/self.CVXPY_1_form_1()[2][1, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
        plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
        plt.plot(xp, ypt, 'k', alpha=0.3, label='true')
        plt.plot(xp, ypt-1, '--k', alpha=0.3)
        plt.plot(xp, ypt+1, '--k', alpha=0.3)
        plt.plot(xp, yp, 'g', linewidth=3, label='CVXPY_1(form 1)')
        plt.title('linearly and strictly separable classes', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1)
        plt.axis('equal')
        plt.xlim([np.min(x1), np.max(x1)])
        plt.ylim([np.min(x2), np.max(x2)])

        fig = plt.show()

        return fig

    def CVXPY_1_form_2(self):

        N = class_1.shape[0]
        M = class_0.shape[0]

        X1 = np.hstack([np.ones([N, 1]), x1[class_1], x2[class_1]])
        X0 = np.hstack([np.ones([M, 1]), x1[class_0], x2[class_0]])

        X1 = np.asmatrix(X1)
        X0 = np.asmatrix(X0)

        w = cvx.Variable([3, 1])

        obj = cvx.Minimize(1)
        const = [X1*w >= 1, X0*w <= -1]
        prob = cvx.Problem(obj, const).solve()

        w = w.value

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        predict = np.sign(V[0, :]*w)[0, 0]

        return predict, w

    def CVXPY_1_form_2_plot(self):

        xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
        yp = - self.CVXPY_1_form_2()[1][1, 0]/self.CVXPY_1_form_2(
        )[1][2, 0]*xp - self.CVXPY_1_form_2()[1][0, 0]/self.CVXPY_1_form_2()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
        plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
        plt.plot(xp, ypt, 'k', alpha=0.3, label='True')
        plt.plot(xp, ypt-1, '--k', alpha=0.3)
        plt.plot(xp, ypt+1, '--k', alpha=0.3)
        plt.plot(xp, yp, 'g', linewidth=3, label='CVXPY_1(form 2)')
        plt.title('linearly and strictly separable classes', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(x1), np.max(x1)])
        plt.ylim([np.min(x2), np.max(x2)])

        fig = plt.show()

        return fig


testing_data = [CVXPY_1(4, -2),
                CVXPY_1(3, 2),
                CVXPY_1(1, 2)]
# ----------------------------------------------------------------------------------------------------
# generate random data

# linearly separable training data + outlier

# if outlier exists -> CVXPY_1: none

# num of data
m = 100

# input(x) -> (x1, x2): 2-dimensional
x1 = 8*np.random.rand(m, 1)
x2 = 7*np.random.rand(m, 1) - 4

# classifier(decision boundary)
g = 0.8*x1 + x2 - 3
g1 = g - 1
g0 = g + 1

# output(y) -> color
class_1 = np.where(g1 >= 0)[0]
class_0 = np.where(g0 < 0)[0]

xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
ypt = -0.8*xp + 3

# outlier
outlier = np.array([[1, 3, 2], [1, 4, 1.5], [1, 6, 0]]).reshape(-1, 3)

X1 = np.hstack([np.ones([class_1.shape[0], 1]), x1[class_1], x2[class_1]])
X0 = np.hstack([np.ones([class_0.shape[0], 1]), x1[class_0], x2[class_0]])

X0 = np.vstack([X0, outlier])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

N = X1.shape[0]
M = X0.shape[0]

# plot
plt.figure(figsize=(10, 8))
plt.plot(X1[:, 1], X1[:, 2], 'ro', alpha=0.4, label='class 1')
plt.plot(X0[:, 1], X0[:, 2], 'bo', alpha=0.4, label='class 0')
plt.title('when outliers exist', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([np.min(x1), np.max(x1)])
plt.ylim([np.min(x2), np.max(x2)])

plt.show()
# ----------------------------------------------------------------------------------------------------
# CVXPY_2: 2nd attempt


class CVXPY_2:

    def __init__(self, v1, v2):

        self.v1 = v1
        self.v2 = v2

    def CVXPY_2_form_2(self):

        w = cvx.Variable([3, 1])
        u = cvx.Variable([N, 1])
        v = cvx.Variable([M, 1])

        obj = cvx.Minimize(np.ones((1, N))*u + np.ones((1, M))*v)
        const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0]
        prob = cvx.Problem(obj, const).solve()

        w = w.value

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        predict = np.sign(V[0, :]*w)[0, 0]

        return predict, w

    def CVXPY_2_form_2_plot(self):

        xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
        yp = - self.CVXPY_2_form_2()[1][1, 0]/self.CVXPY_2_form_2(
        )[1][2, 0]*xp - self.CVXPY_2_form_2()[1][0, 0]/self.CVXPY_2_form_2()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(X1[:, 1], X1[:, 2], 'ro', alpha=0.4, label='class 1')
        plt.plot(X0[:, 1], X0[:, 2], 'bo', alpha=0.4, label='class 0')
        plt.plot(xp, ypt, 'k', alpha=0.3, label='True')
        plt.plot(xp, ypt-1, '--k', alpha=0.3)
        plt.plot(xp, ypt+1, '--k', alpha=0.3)
        plt.plot(xp, yp, 'g', linewidth=3, label='CVXPY_2(form 2)')
        plt.plot(xp, yp-1/self.CVXPY_2_form_2()[1][2, 0], '--g')
        plt.plot(xp, yp+1/self.CVXPY_2_form_2()[1][2, 0], '--g')
        plt.title('when outliers exist', fontsize=15)
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(x1), np.max(x1)])
        plt.ylim([np.min(x2), np.max(x2)])

        fig = plt.show()

        return fig


testing_data = [CVXPY_2(5.5, -0.3),
                CVXPY_2(3, 2),
                CVXPY_2(1, 2)]
# ----------------------------------------------------------------------------------------------------
# SVM: support vector machine


class SVM:

    def __init__(self, v1, v2, gamma):

        self.v1 = v1
        self.v2 = v2
        self.gamma = gamma

    def SVM_form_2(self):

        w = cvx.Variable([3, 1])
        u = cvx.Variable([N, 1])
        v = cvx.Variable([M, 1])

        obj = cvx.Minimize(cvx.norm(w, 2) + self.gamma *
                           (np.ones((1, N))*u + np.ones((1, M))*v))
        const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0]
        prob = cvx.Problem(obj, const).solve()

        w = w.value

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        predict = np.sign(V[0, :]*w)[0, 0]

        return predict, w

    def SVM_form_2_plot(self):

        xp = np.linspace(np.min(x1), np.max(x1), 100).reshape(-1, 1)
        yp = - self.SVM_form_2()[1][1, 0]/self.SVM_form_2()[1][2, 0] * \
            xp - self.SVM_form_2()[1][0, 0]/self.SVM_form_2()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(X1[:, 1], X1[:, 2], 'ro', alpha=0.4, label='class 1')
        plt.plot(X0[:, 1], X0[:, 2], 'bo', alpha=0.4, label='class 0')
        plt.plot(xp, ypt, 'k', alpha=0.3, label='True')
        plt.plot(xp, ypt-1, '--k', alpha=0.3)
        plt.plot(xp, ypt+1, '--k', alpha=0.3)
        plt.plot(xp, yp, 'g', linewidth=3, label='SVM(form 2)')
        plt.title('when outliers exist', fontsize=15)
        plt.plot(xp, yp-1/self.SVM_form_2()[1][2, 0], '--g')
        plt.plot(xp, yp+1/self.SVM_form_2()[1][2, 0], '--g')
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(x1), np.max(x1)])
        plt.ylim([np.min(x2), np.max(x2)])

        fig = plt.show()

        return fig


testing_data = [SVM(5.5, -0.3, 2),
                SVM(3, 2, 2),
                SVM(1, 2, 2)]
# ====================================================================================================
