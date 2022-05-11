# ====================================================================================================
# RF: random forest
# ====================================================================================================
# summary about random forest

# feature test
# homogenous set
# ----------------------------------------------------------------------------------------------------
# import library

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import pydotplus
from sklearn import ensemble
from sklearn import neighbors
from sklearn import tree
from sklearn.tree import export_graphviz
from IPython.display import Image
# ----------------------------------------------------------------------------------------------------
# generate sample data -> three simulated clusters

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
# scikit random forest


def random_forest(v1, v2, num, depth):

    clf = ensemble.RandomForestClassifier(
        n_estimators=num, max_depth=depth, random_state=0)
    clf.fit(X, np.ravel(y))

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
