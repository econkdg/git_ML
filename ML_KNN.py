# ====================================================================================================
# KNN(k-nearest neighbor): non-parametric
# ====================================================================================================
# summary about KNN

# 1. non-parametric -> data-driven
# 2. regression or classification problem
# 3. design tuning parameter(k) -> KNN performance
# ----------------------------------------------------------------------------------------------------
# import library

from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------
# generate random data(1)

N = 100

w1 = 0.5
w0 = 2

x = np.random.normal(0, 15, N).reshape(-1, 1)
y = w1*x + w0 + 5*np.random.normal(0, 1, N).reshape(-1, 1)

plt.figure(figsize=(10, 8))
plt.title('data set', fontsize=15)
plt.plot(x, y, '.', label='data')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.axis('equal')
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.grid(alpha=0.3)

plt.show()
# ----------------------------------------------------------------------------------------------------
# KNN regression: non-parametric

# scikit learn


def KNN_regression(point_new, n_neighbors):

    reg = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(x, y)

    x_new = np.array([[point_new]])

    predict = reg.predict(x_new)[0, 0]

    xp = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1)
    yp = reg.predict(xp)

    plt.figure(figsize=(10, 8))
    plt.title('k-nearest neighbor regression', fontsize=15)
    plt.plot(x, y, '.', label='original data')
    plt.plot(xp, yp, color='r', label='KNN')
    plt.plot(x_new, predict, 'o', color='k', label='prediction')
    plt.plot([x_new[0, 0], x_new[0, 0]], [
             np.min(y)-10, predict], 'k--', alpha=0.5)
    plt.plot([np.min(x)-5, x_new[0, 0]], [predict, predict], 'k--', alpha=0.5)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.legend(fontsize=12)
    plt.axis('equal')
    plt.axis([np.min(x)-5, np.max(x)+5, np.min(y)-5, np.max(y)+5])
    plt.grid(alpha=0.3)

    fig = plt.show()

    return predict, fig
# ----------------------------------------------------------------------------------------------------
# generate random data(2)


m = 1000

X = -1.5 + 3*np.random.uniform(size=(m, 2))
y = np.zeros([m, 1])

for i in range(m):

    if np.linalg.norm(X[i, :], 2) <= 1:
        y[i] = 1

    else:
        y[i] = 0

class_1 = np.where(y == 1)[0]
class_0 = np.where(y == 0)[0]

theta = np.linspace(0, 2*np.pi, 100)

plt.figure(figsize=(8, 8))
plt.plot(X[class_1, 0], X[class_1, 1], 'o', label='class 1',
         markerfacecolor="k", markeredgecolor='k', markersize=4)
plt.plot(X[class_0, 0], X[class_0, 1], 'o', label='class 0', markerfacecolor="none",
         alpha=0.3, markeredgecolor='k', markersize=4)
plt.plot(np.cos(theta), np.sin(theta), '--', color='orange')
plt.axis([np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1])])
plt.axis('equal')
plt.axis('off')

plt.show()
# ----------------------------------------------------------------------------------------------------
# generate random data(3): outliers


m = 1000

X = -1.5 + 3*np.random.uniform(size=(m, 2))
y = np.zeros([m, 1])

for i in range(m):

    if np.linalg.norm(X[i, :], 2) <= 1:

        if np.random.uniform() < 0.05:
            y[i] = 0

        else:
            y[i] = 1

    else:
        if np.random.uniform() < 0.05:
            y[i] = 1

        else:
            y[i] = 0

class_1 = np.where(y == 1)[0]
class_0 = np.where(y == 0)[0]

theta = np.linspace(0, 2*np.pi, 100)

plt.figure(figsize=(8, 8))
plt.plot(X[class_1, 0], X[class_1, 1], 'o', label='class 1',
         markerfacecolor='k', markeredgecolor='k', markersize=4)
plt.plot(X[class_0, 0], X[class_0, 1], 'o', label='class 0',
         markerfacecolor='none', alpha=0.3, markeredgecolor='k', markersize=4)
plt.plot(np.cos(theta), np.sin(theta), '--', color='orange')
plt.axis([np.min(X[:, 0]), np.max(X[:, 0]), np.min(X[:, 1]), np.max(X[:, 1])])
plt.axis('equal')
plt.axis('off')
plt.show()
# ----------------------------------------------------------------------------------------------------
# KNN classification: non-parametric

# scikit learn


def KNN_classification(v1, v2, n_neighbors):

    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, np.ravel(y))

    X_new = np.array([v1, v2]).reshape(1, -1)

    predict = clf.predict(X_new)[0]

    res = 0.01

    [X1gr, X2gr] = np.meshgrid(np.arange(np.min(X[:, 0]), np.max(
        X[:, 0]), res), np.arange(np.min(X[:, 1]), np.max(X[:, 1]), res))

    Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
    Xp = np.asmatrix(Xp)

    in_class_1 = clf.predict(Xp).reshape(-1, 1)
    in_circle = np.where(in_class_1 == 1)[0]

    plt.figure(figsize=(8, 8))
    plt.plot(X[class_1, 0], X[class_1, 1], 'o', label='class 1', markerfacecolor="k",
             alpha=0.5, markeredgecolor='k', markersize=4)
    plt.plot(X[class_0, 0], X[class_0, 1], 'o', label='class 0', markerfacecolor="none",
             alpha=0.3, markeredgecolor='k', markersize=4)
    plt.plot(np.cos(theta), np.sin(theta), '--', color='orange')
    plt.plot(Xp[in_circle][:, 0], Xp[in_circle][:, 1],
             's', alpha=0.5, color='g', markersize=1)
    plt.plot(v1, v2, 'o', color='r', label='testing data')
    plt.axis([np.min(X[:, 0]), np.max(X[:, 0]),
             np.min(X[:, 1]), np.max(X[:, 1])])
    plt.axis('equal')
    plt.axis('off')

    fig = plt.show()

    return predict, fig
# ====================================================================================================
