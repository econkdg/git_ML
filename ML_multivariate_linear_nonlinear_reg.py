# ====================================================================================================
# multivariate linear & nonlineaer regression
# ====================================================================================================
# multivariate linear regression

# model: y = theta0 + theta1*x1 + theta2*x2 + noise

# 3D plot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# generate random data
n = 200
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
noise = 0.5*np.random.randn(n, 1)

y = 2 + 1*x1 + 3*x2 + noise

# plot
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('generated data', fontsize=15)
ax.set_xlabel('$X_1$', fontsize=15)
ax.set_ylabel('$X_2$', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
ax.scatter(x1, x2, y, marker='.', label='data')
# ax.view_init(30,30)

plt.legend(fontsize=15)
plt.show()

# estimation
# % matplotlib qt5
A = np.hstack([np.ones((n, 1)), x1, x2])
A = np.asmatrix(A)

theta = (A.T*A).I*A.T*y

X1, X2 = np.meshgrid(np.arange(np.min(x1), np.max(x1), 0.1),
                     np.arange(np.min(x2), np.max(x2), 0.1))

YP = theta[0, 0] + theta[1, 0]*X1 + theta[2, 0]*X2

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('multivariate linear regression', fontsize=15)
ax.set_xlabel('$X_1$', fontsize=15)
ax.set_ylabel('$X_2$', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
ax.scatter(x1, x2, y, marker='.', label='data')
ax.plot_wireframe(X1, X2, YP, color='k', alpha=0.3, label='regression plane')
# ax.view_init(30,30)

plt.legend(fontsize=15)
plt.show()
# ----------------------------------------------------------------------------------------------------
# nonlinear regression

# method: linear combination of basis functions -> target function
# (1) polynomial basis function(PBF)
# (2) radial basis function(RBF)

# PBF
xp = np.arange(-1, 1, 0.01).reshape(-1, 1)
pbfbasis = np.hstack([xp**i for i in range(6)])  # 0 ~ 5

plt.figure(figsize=(10, 8))

for i in range(6):
    plt.plot(xp, pbfbasis[:, i], label='$x^{}$'.format(i))

plt.title('pbf basis', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.show()

# RBF(bell shape -> localized fitting)
d = 9
u = np.linspace(-1, 1, d)  # RBF center
sigma = 0.1  # kutosis

xp = np.arange(-1, 1, 0.01).reshape(-1, 1)
rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])

for i in range(d):
    plt.plot(xp, rbfbasis[:, i], label='$\mu = {}$'.format(u[i]))

plt.title('rbf basis', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.show()
# ----------------------------------------------------------------------------------------------------
# model: y = theta0 + theta1*x + theta2*x^2 + noise

# generate random data
n = 100
x = -5 + 15*np.random.rand(n, 1)  # uniform: [-5, 10]
noise = 10*np.random.randn(n, 1)  # Gaussian

y = 10 + 1*x + 2*x**2 + noise

# plot
plt.figure(figsize=(10, 8))

plt.title('True x and y', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'o', markersize=4, label='actual')
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()

# special estimation(2nd-degree PBF)
A = np.hstack([x**0, x, x**2])
A = np.asmatrix(A)

theta = (A.T*A).I*A.T*y

xp = np.linspace(np.min(x), np.max(x))  # linspace default -> 50 1d array
yp = theta[0, 0] + theta[1, 0]*xp + theta[2, 0]*xp**2

plt.figure(figsize=(10, 8))

plt.plot(x, y, 'o', markersize=4, label='actual')
plt.plot(xp, yp, 'r', linewidth=2, label='estimated')

plt.title('nonlinear regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()
# ----------------------------------------------------------------------------------------------------
# general estimation(nth-degree PBF & RBF)

# generate random data
n = 100
x = -5 + 15*np.random.rand(n, 1)  # uniform: [-5, 10]
noise = 10*np.random.randn(n, 1)  # Gaussian

y = 10 + 1*x + 2*x**2 + noise

# PBF
xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)


def polynomial_basis(m):
    pbfbasis = np.hstack([xp**i for i in range(m)])
    pbfbasis = np.asmatrix(pbfbasis)

    A = np.hstack([x**i for i in range(m)])
    A = np.asmatrix(A)

    theta = (A.T*A).I*A.T*y
    return pbfbasis, theta


pbfbasis, theta = polynomial_basis(3)
yp = pbfbasis*theta

plt.figure(figsize=(10, 8))

plt.plot(x, y, 'o', markersize=4, label='actual')
plt.plot(xp, yp, 'r', linewidth=2, label='PBF')

plt.title('nonlinear regression with PBF basis', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()

# RBF
xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

d = 6
u = np.linspace(np.min(x), np.max(x), d)


def radial_basis(m):
    sigma = m  # tunning parameter

    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2))
                         for i in range(d)])
    rbfbasis = np.asmatrix(rbfbasis)

    A = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
    A = np.asmatrix(A)

    theta = (A.T*A).I*A.T*y
    return rbfbasis, theta


rbfbasis, theta = radial_basis(4)
yp = rbfbasis*theta

plt.figure(figsize=(10, 8))

plt.plot(x, y, 'o', markersize=4, label='actual')
plt.plot(xp, yp, 'r', linewidth=2, label='RBF')

plt.title('nonlinear regression with RBF basis', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()
# ----------------------------------------------------------------------------------------------------
# multivariate & nonlinear regression

# model: y = theta0 + theta1*x1 + theta2*x1^2 + theta3*x2 + theta4*x2^2 + noise

# generate random data
n = 1000
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
noise = 0.5*np.random.randn(n, 1)

y = 2 + 1*x1 + 3*x1**2 + 2*x2 + 5*x2**2 + noise

# plot
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('generated data', fontsize=15)
ax.set_xlabel('$X_1$', fontsize=15)
ax.set_ylabel('$X_2$', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
ax.scatter(x1, x2, y, marker='.', label='data')
# ax.view_init(30,30)

plt.legend(fontsize=15)
plt.show()

# estimation
# % matplotlib qt5
A = np.hstack([np.ones((n, 1)), x1, x1**2, x2, x2**2])
A = np.asmatrix(A)

theta = (A.T*A).I*A.T*y

X1, X2 = np.meshgrid(np.arange(np.min(x1), np.max(x1), 0.1),
                     np.arange(np.min(x2), np.max(x2), 0.1))

YP = theta[0, 0] + theta[1, 0]*X1 + theta[2, 0] * \
    X1**2 + theta[3, 0]*X2 + theta[4, 0]*X2**2

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title(
    'method1: constructing explicit feature vectors -> polynomial feature', fontsize=15)
ax.set_xlabel('$X_1$', fontsize=15)
ax.set_ylabel('$X_2$', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
ax.scatter(x1, x2, y, marker='.', color='k', label='data')
ax.plot_wireframe(X1, X2, YP, color='k', alpha=0.3, label='hyper plane')
ax.plot_surface(X1, X2, YP, rstride=4, cstride=4, alpha=0.4, cmap='jet')
# ax.view_init(30,30)

plt.legend(fontsize=15)
plt.show()
# ====================================================================================================
