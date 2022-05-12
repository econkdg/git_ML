# ====================================================================================================
# linear regression
# ====================================================================================================
# import library

from sklearn import linear_model
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------
# generate data(input, output)

# 10 data points(column vector)
x = np.array([0.1, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8,
             3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1, 1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8,
             2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1, 1)

# generate random data
n = 100
x = np.random.randn(n, 1)
noise = np.random.randn(n, 1)

y = 2 + 1.5*x + noise

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko')
plt.title('Data', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# linear algebra method

# LR function(algorithm: least square)


def LR(n):
    x = np.random.randn(n, 1)
    noise = np.random.randn(n, 1)

    y = 2 + 1.5*x + noise

    m = y.shape[0]
    #A = np.hstack([np.ones([m, 1]), x])
    A = np.hstack([x**0, x])
    A = np.asmatrix(A)

    theta = (A.T*A).I*A.T*y

    return theta


print('theta:\n', LR(100))

# plot
plt.figure(figsize=(10, 8))
plt.title('Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko', label="data")

# fitted line
xp = np.arange(-3, 3, 0.01).reshape(-1, 1)
yp = LR(100)[0, 0] + LR(100)[1, 0]*xp

plt.plot(xp, yp, 'r', linewidth=2, label="regression")
plt.legend(fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# gradient descent

# generate random data
n = 100
x = np.random.randn(n, 1)
noise = np.random.randn(n, 1)

y = 2 + 1.5*x + noise

# gradient function
A = np.hstack([x**0, x])
A = np.asmatrix(A)

alpha = 0.001


def gradient(n, m):
    THETA = []

    for i in range(n):
        theta = np.random.randn(2, 1)
        theta = np.asmatrix(theta)

        for i in range(m):

            df = 2*(A.T*A*theta - A.T*y)
            theta = theta - alpha*df

        THETA.append(theta)

    np.mean(THETA, axis=0)

    return THETA


print(gradient(10, 100))
# ----------------------------------------------------------------------------------------------------
# CVXPY

# generate random data
n = 100
x = np.random.randn(n, 1)
noise = np.random.randn(n, 1)

y = 2 + 1.5*x + noise

# l_n-norm
A = np.hstack([x**0, x])
A = np.asmatrix(A)

alpha = 0.001


def cvxpy(n):

    theta = cvx.Variable([2, 1])
    obj = cvx.Minimize(cvx.norm(A*theta-y, n))
    cvx.Problem(obj, []).solve()

    theta = theta.value

    return theta


# plot
plt.figure(figsize=(10, 8))
plt.title('$L_1$ and $L_2$ Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko', label='data')

# fitted line
xp = np.arange(-3, 3, 0.01).reshape(-1, 1)
yp1 = cvxpy(1)[0, 0] + cvxpy(1)[1, 0]*xp
yp2 = cvxpy(2)[0, 0] + cvxpy(2)[1, 0]*xp

plt.plot(xp, yp1, 'b', linewidth=2, label='$L_1$')
plt.plot(xp, yp2, 'r', linewidth=2, label='$L_2$')
plt.legend(fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# outlier
x = np.vstack([x, np.array([3.0, 10.0]).reshape(-1, 1)])
y = np.vstack([y, np.array([10.0, 3.0]).reshape(-1, 1)])

A = np.hstack([x**0, x])
A = np.asmatrix(A)

A.shape

# plot
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko', label='data')
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# cvxpy(2)
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko', label='data')
xp = np.arange(-5, 5, 0.01).reshape(-1, 1)
yp2 = cvxpy(2)[0, 0] + cvxpy(2)[1, 0]*xp

plt.plot(xp, yp2, 'r', linewidth=2, label='$L_2$')
plt.axis('equal')
plt.legend(fontsize=15, loc=5)
plt.grid(alpha=0.3)
plt.show()

# cvxpy(1)
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko', label='data')
xp = np.arange(-5, 5, 0.01).reshape(-1, 1)
yp1 = cvxpy(1)[0, 0] + cvxpy(1)[1, 0]*xp

plt.plot(xp, yp1, 'b', linewidth=2, label='$L_1$')
plt.axis('equal')
plt.legend(fontsize=15, loc=5)
plt.grid(alpha=0.3)
plt.show()

# plot
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko', label='data')
plt.title('$L_1$ and $L_2$ Regression w/ Outliers', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)

# fitted line
xp = np.arange(-5, 5, 0.01).reshape(-1, 1)
yp1 = cvxpy(1)[0, 0] + cvxpy(1)[1, 0]*xp
yp2 = cvxpy(2)[0, 0] + cvxpy(2)[1, 0]*xp

plt.plot(xp, yp1, 'b', linewidth=2, label='$L_1$')
plt.plot(xp, yp2, 'r', linewidth=2, label='$L_2$')
plt.axis('scaled')
plt.legend(fontsize=15, loc=5)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# scikit learn

# generate data
x = np.array([0.1, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8,
             3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1, 1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8,
             2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1, 1)

# linear regression
reg = linear_model.LinearRegression()
reg.fit(x, y)

reg.coef_
reg.intercept_

# plot
plt.figure(figsize=(10, 8))
plt.title('Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko', label="data")

# fitted line
plt.plot(xp, reg.predict(xp), 'r', linewidth=2, label="regression")
plt.legend(fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()
# ====================================================================================================
