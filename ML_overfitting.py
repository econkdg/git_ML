# ====================================================================================================
# overfitting
# ====================================================================================================
# PBF

# import library
import numpy as np
import matplotlib.pyplot as plt

# generate data
n = 10
x = np.linspace(-4.5, 4.5, 10).reshape(-1, 1)
y = np.array([0.9819, 0.7973, 1.9737, 0.1838, 1.3180, -0.8361, -
             0.6591, -2.4701, -2.8122, -6.2512]).reshape(-1, 1)

# plot
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 1st degree
A = np.hstack([x**0, x])
A = np.asmatrix(A)

theta = (A.T*A).I*A.T*y
print(theta)

xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)
yp = theta[0, 0] + theta[1, 0]*xp

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='data')
plt.plot(xp[:, 0], yp[:, 0], linewidth=2, label='linear')
plt.title('linear regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 2nd degree
A = np.hstack([x**0, x, x**2])
A = np.asmatrix(A)

theta = (A.T*A).I*A.T*y
print(theta)

xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)
yp = theta[0, 0] + theta[1, 0]*xp + theta[2, 0]*xp**2

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='data')
plt.plot(xp[:, 0], yp[:, 0], linewidth=2, label='2nd degree')
plt.title('nonlinear regression with polynomial functions', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 9th degree(# of unknown = # of equation)


def polynomial_basis(m):
    A = np.hstack([x**i for i in range(m+1)])
    A = np.asmatrix(A)

    theta = (A.T*A).I*A.T*y

    xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)

    pbfbasis = np.hstack([xp**i for i in range(m+1)])
    pbfbasis = np.asmatrix(pbfbasis)

    return pbfbasis, theta


pbfbasis, theta = polynomial_basis(9)
yp = pbfbasis*theta

plt.figure(figsize=(10, 8))

plt.plot(x, y, 'o', label='Data')
plt.plot(xp[:, 0], yp[:, 0], linewidth=2, label='9th degree')

plt.title('nonlinear regression with pbf basis', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# model: y = theta0 + theta1*x + theta2*x^2 + theta3*x^3 + noise

# generate random data
n = 100
x = -5 + 10*np.random.rand(n, 1)  # uniform: [-5, 10]
noise = 10*np.random.randn(n, 1)  # Gaussian

y = 10 + 0.70*x + 0.35*(x-5)**2 + 0.25*x**3 + noise


# PBF
xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

d = [1, 3, 5, 9]

RSS = []

plt.figure(figsize=(12, 10))
plt.suptitle('nonlinear regression', fontsize=15)

for k in range(4):  # 0 ~ 3
    A = np.hstack([x**i for i in range(d[k]+1)])  # 0 ~ d[k]
    pbfbasis = np.hstack([xp**i for i in range(d[k]+1)])

    A = np.asmatrix(A)
    pbfbasis = np.asmatrix(pbfbasis)

    theta = (A.T*A).I*A.T*y
    yp = pbfbasis*theta

    RSS.append(np.linalg.norm(y - A*theta, 2)**2)

    plt.subplot(2, 2, k+1)
    plt.plot(x, y, 'o', color='k', markersize=3, label='actual')
    plt.plot(xp, yp, color='r', label='PBF')
    plt.axis([-5, 5, -50, 50])
    plt.title('degree = {}'.format(d[k]))
    plt.grid(alpha=0.3)

plt.show()

# RSS
plt.figure(figsize=(10, 8))
plt.stem(d, RSS, label='RSS')
plt.title('residual sum of squares', fontsize=15)
plt.xlabel('degree', fontsize=15)
plt.ylabel('RSS', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# RBF
xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

d = [3, 6, 9, 12, 15, 18]


def rbf_sigma(k, m):

    u = np.linspace(-5, 5, d[k])

    A = np.hstack([np.exp(-(x-u[i])**2/(2*m**2)) for i in range(d[k])])
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*m**2))
                          for i in range(d[k])])

    A = np.asmatrix(A)
    rbfbasis = np.asmatrix(rbfbasis)

    theta = (A.T*A).I*A.T*y

    return rbfbasis, theta


plt.figure(figsize=(12, 10))

for k in range(6):

    rbfbasis, theta = rbf_sigma(k, 1)
    yp = rbfbasis*theta

    plt.subplot(2, 3, k+1)
    plt.plot(x, y, 'o', color='k', markersize=3, label='actual')
    plt.plot(xp, yp, color='r', label='RBF')
    plt.axis([-5, 5, -50, 50])
    plt.title('num RBFs = {}'.format(d[k]), fontsize=10)
    plt.grid(alpha=0.3)

    plt.suptitle('nonlinear regression with RBF', fontsize=15)

plt.show()
# ====================================================================================================
