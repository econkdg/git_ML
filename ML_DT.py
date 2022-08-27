# ====================================================================================================
# DT: decision tree
# ====================================================================================================
# summary about decision tree

# feature test
# homogenous set
# ----------------------------------------------------------------------------------------------------
# import library

import sys
import os
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
import pydotplus
from IPython.display import Image
# ----------------------------------------------------------------------------------------------------
# disorder <=> entropy

# information theory

x = np.linspace(0, 1, 100)
y = -x*np.log2(x) - (1-x)*np.log2(1-x)

plt.figure(figsize=(10, 8))
plt.plot(x, y, linewidth=3)
plt.xlabel(r'$x$', fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)

plt.show()
# ----------------------------------------------------------------------------------------------------
# generate sample data(1)


data = np.array([[0, 0, 1, 0, 0],
                 [1, 0, 2, 0, 0],
                 [0, 1, 2, 0, 1],
                 [2, 1, 0, 2, 1],
                 [0, 1, 0, 1, 1],
                 [1, 1, 1, 2, 0],
                 [1, 1, 0, 2, 0],
                 [0, 0, 2, 1, 0]])

x = data[:, 0:4]
y = data[:, 4]
# ----------------------------------------------------------------------------------------------------
# quality of test


def disorder(x):

    y = -x*np.log2(x) - (1-x)*np.log2(1-x)

    return y
# ----------------------------------------------------------------------------------------------------
# scikit decision tree(1)

# decision tree -> only binary


def decision_tree(feature_1, feature_2, feature_3, feature_4, depth):

    clf = tree.DecisionTreeClassifier(
        criterion='entropy', max_depth=depth, random_state=0)
    clf.fit(x, y)  # random_state: random gererate, random seed

    predict = clf.predict([[feature_1, feature_2, feature_3, feature_4]])

    fig = tree.plot_tree(clf)

    dot_data = tree.export_graphviz(clf)
    graph = graphviz.Source(dot_data)

    graph.render("decision tree")  # render file as pdf

    return predict, fig, graph
# ----------------------------------------------------------------------------------------------------
# generate sample data(2)


class_1 = np.array([[-1.1, 0], [-0.3, 0.1], [-0.9, 1], [0.8, 0.4], [0.4, 0.9], [0.3, -0.6],
                    [-0.5, 0.3], [-0.8, 0.6], [-0.5, -0.5]])
class_0 = np.array([[-1, -1.3], [-1.6, 2.2], [0.9, -0.7], [1.6, 0.5], [1.8, -1.1], [1.6, 1.6],
                    [-1.6, -1.7], [-1.4, 1.8], [1.6, -0.9], [0, -1.6], [0.3, 1.7], [-1.6, 0], [-2.1, 0.2]])

class_1 = np.asmatrix(class_1)
class_0 = np.asmatrix(class_0)

N = class_1.shape[0]
M = class_0.shape[0]

X = np.vstack([class_1, class_0])
y = np.vstack([np.ones([N, 1]), np.zeros([M, 1])])

plt.figure(figsize=(10, 8))
plt.plot(class_1[:, 0], class_1[:, 1], 'ro', label='class 1')
plt.plot(class_0[:, 0], class_0[:, 1], 'bo', label='class 0')
plt.title('nonlinear data', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')

plt.show()
# ----------------------------------------------------------------------------------------------------
# scikit decision tree(2)


def decision_tree(v1, v2, depth):

    clf = tree.DecisionTreeClassifier(
        criterion='entropy', max_depth=depth, random_state=0)
    clf.fit(X, y)

    predict = clf.predict([[v1, v2]])

    [X1gr, X2gr] = np.meshgrid(
        np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.1), np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.1))

    Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
    Xp = np.asmatrix(Xp)

    q = clf.predict(Xp)
    q = np.asmatrix(q).reshape(-1, 1)

    c1 = np.where(q == 1)[0]

    plt.figure(figsize=(10, 8))
    plt.plot(class_1[:, 0], class_1[:, 1], 'ro', label='class 1')
    plt.plot(class_0[:, 0], class_0[:, 1], 'bo', label='class 0')
    plt.plot(Xp[c1, 0], Xp[c1, 1], 'gs',
             markersize=8, alpha=0.1, label='decision tree')
    plt.plot(v1, v2, 'o', color='k', label='testing data')
    plt.xlabel(r'$x_1$', fontsize=15)
    plt.ylabel(r'$x_2$', fontsize=15)
    plt.legend(loc=1, fontsize=12)
    plt.axis('equal')

    fig = plt.show()

    return predict, fig
# ----------------------------------------------------------------------------------------------------
# generate sample data(3) -> three simulated clusters


mu1 = np.array([1, 7])
mu2 = np.array([3, 4])
mu3 = np.array([6, 5])

sigma1 = 0.8*np.array([[1, 1.5],
                       [1.5, 3]])
sigma2 = 0.5*np.array([[2, 0],
                       [0, 2]])
sigma3 = 0.5*np.array([[1, -1],
                       [-1, 2]])

X1 = np.random.multivariate_normal(mu1, sigma1, 100)
X2 = np.random.multivariate_normal(mu2, sigma2, 100)
X3 = np.random.multivariate_normal(mu3, sigma3, 100)

y1 = 1*np.ones([100, 1])
y2 = 2*np.ones([100, 1])
y3 = 3*np.ones([100, 1])

X = np.vstack([X1, X2, X3])
y = np.vstack([y1, y2, y3])

plt.figure(figsize=(10, 8))
plt.title('generated data', fontsize=15)
plt.plot(X1[:, 0], X1[:, 1], '.', label='class 1')
plt.plot(X2[:, 0], X2[:, 1], '.', label='class 2')
plt.plot(X3[:, 0], X3[:, 1], '.', label='class 3')
plt.xlabel('$X_1$', fontsize=15)
plt.ylabel('$X_2$', fontsize=15)
plt.legend(fontsize=12)
plt.axis('equal')
plt.grid(alpha=0.3)

plt.show()
# ----------------------------------------------------------------------------------------------------
# scikit decision tree(3)


def decision_tree(v1, v2, depth):

    clf = tree.DecisionTreeClassifier(
        criterion='entropy', max_depth=depth, random_state=0)
    clf.fit(X, y)

    predict = clf.predict([[v1, v2]])

    [X1gr, X2gr] = np.meshgrid(
        np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.1), np.arange(np.min(X[:, 1]), np.max(X[:, 1]), 0.1))

    Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
    Xp = np.asmatrix(Xp)

    q = clf.predict(Xp)
    q = np.asmatrix(q).reshape(-1, 1)

    c1 = np.where(q == 1)[0]
    c2 = np.where(q == 2)[0]
    c3 = np.where(q == 3)[0]

    plt.figure(figsize=(10, 8))
    plt.plot(X1[:, 0], X1[:, 1], '.', label='class 1')
    plt.plot(X2[:, 0], X2[:, 1], '.', label='class 2')
    plt.plot(X3[:, 0], X3[:, 1], '.', label='class 3')
    plt.plot(Xp[c1, 0], Xp[c1, 1], 's', color='blue', markersize=8, alpha=0.05)
    plt.plot(Xp[c2, 0], Xp[c2, 1], 's',
             color='orange', markersize=8, alpha=0.05)
    plt.plot(Xp[c3, 0], Xp[c3, 1], 's',
             color='green', markersize=8, alpha=0.05)
    plt.plot(v1, v2, 'o', color='k', label='testing data')
    plt.xlabel('$X_1$', fontsize=15)
    plt.ylabel('$X_2$', fontsize=15)
    plt.legend(fontsize=12)
    plt.axis('equal')
    plt.grid(alpha=0.3)

    fig = plt.show()

    return predict, fig
# ====================================================================================================
