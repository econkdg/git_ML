# ====================================================================================================
# PCA
# ====================================================================================================
# PCA algorithm(2D)

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# generate random data
m = 5000
mu = np.array([0, 0])
sigma = np.array([[3, 1.5],
                  [1.5, 1]])

# pre-processing
X = np.random.multivariate_normal(mu, sigma, m)
X = np.asmatrix(X)

fig = plt.figure(figsize=(10, 8))
plt.plot(X[:, 0], X[:, 1], 'k.', alpha=0.3)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# variance of projected data
S = 1/(m-1)*X.T*X  # sample covariance matrix

D, U = np.linalg.eig(S)  # eigenvalue, eigenvector

idx = np.argsort(-D)  # sorting

D = D[idx]
U = U[:, idx]

# optimization form
# max u.T*S*u (variance of projected data)
# s.t. u.T*u = 1

# eigenvector u
h = U[1, 0]/U[0, 0]
xp = np.arange(-6, 6, 0.1)
yp = h*xp

fig = plt.figure(figsize=(10, 8))
plt.plot(X[:, 0], X[:, 1], 'k.', alpha=0.3)
plt.plot(xp, yp, 'r', linewidth=3)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# projected -> histogram
# u1
Z = X*U[:, 0]

plt.figure(figsize=(10, 8))
plt.hist(Z, 51, color='g')
plt.axis([-10, 10, 0, 300])
plt.show()

# u2
Z = X*U[:, 1]

plt.figure(figsize=(10, 8))
plt.hist(Z, 51, color='g')
plt.axis([-10, 10, 0, 300])
plt.show()
# ----------------------------------------------------------------------------------------------------
# Scikit-learn


def pca(k):
    pca = PCA(n_components=k)

    fit = pca.fit(X)
    LT = pca.transform(X)

    return fit, LT


fit, LT = pca(1)

plt.figure(figsize=(10, 8))
plt.hist(LT, 51, color='g')
plt.show()
# ----------------------------------------------------------------------------------------------------
# PCA algorithm(4D -> 3D)

# generate random data
m = 3000
mu = np.array([0, 0, 0, 0])
sigma = np.array([[3, 1, 2, 2],
                  [1, 5, 1, 2],
                  [2, 1, 1000, 2000],
                  [2, 2, 2000, 1000]])

# pre-processing
X = np.random.multivariate_normal(mu, sigma, m)
X = np.asmatrix(X)

# PCA(2)


def pca(k):
    pca = PCA(n_components=k)

    fit = pca.fit(X)
    LT = pca.transform(X)

    return fit, LT


fit, LT = pca(3)

# plot
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('nth PCA analysis', fontsize=15)
ax.set_xlabel('$LT_1$', fontsize=15)
ax.set_ylabel('$LT_2$', fontsize=15)
ax.set_zlabel('$LT_3$', fontsize=15)
ax.scatter(LT[:, 0], LT[:, 1], LT[:, 2], marker='.', color='r', label='PC(3)')
# ax.view_init(30,30)

plt.legend(fontsize=15)
plt.show()
# ====================================================================================================
