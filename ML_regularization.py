# ====================================================================================================
# regularization
# ====================================================================================================
# model: y = theta0 + theta1*x + theta2*x^2 + theta3*x^3 + noise

# import library
from sklearn import linear_model
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt

# generate random data
n = 100
x = -5 + 10*np.random.rand(n, 1)  # uniform: [-5, 10]
noise = 5*np.random.randn(n, 1)  # Gaussian

y = 10 + 0.70*x + 0.35*(x-5)**2 + 0.25*x**3 + noise

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

# cvxpy


def rbf_cvxpy(d, m):

    u = np.linspace(np.min(x), np.max(x), d)

    A = np.hstack([np.exp(-(x-u[i])**2/(2*m**2)) for i in range(d)])
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*m**2))
                         for i in range(d)])

    A = np.asmatrix(A)
    rbfbasis = np.asmatrix(rbfbasis)

    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta-y))
    prob = cvx.Problem(obj)
    result = prob.solve()

    return rbfbasis, theta


rbfbasis, theta = rbf_cvxpy(15, 1)
yp = rbfbasis*theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', color='k', label='data')
plt.plot(xp, yp, color='r', label='overfitted')
plt.title('(overfitted) regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# ridge regression


def ridge(d, m, lamb):

    u = np.linspace(np.min(x), np.max(x), d)

    A = np.hstack([np.exp(-(x-u[i])**2/(2*m**2)) for i in range(d)])
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*m**2))
                         for i in range(d)])

    A = np.asmatrix(A)
    rbfbasis = np.asmatrix(rbfbasis)

    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) +
                       lamb*cvx.sum_squares(theta))
    prob = cvx.Problem(obj)
    result = prob.solve()

    return rbfbasis, theta


rbfbasis, theta = ridge(15, 1, 10)
yp = rbfbasis*theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', color='k', label='data')
plt.plot(xp, yp, label='ridge')
plt.title('ridge regularization(L2)', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# plot(ridge)

# plot 1
plt.figure(figsize=(10, 8))
plt.title(r'ridge: magnitude of $\theta$', fontsize=15)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel('magnitude', fontsize=15)
plt.stem(np.linspace(1, 15, 15).reshape(-1, 1), theta.value)
plt.xlim([0, 16])
plt.ylim([np.min(theta.value), np.max(theta.value)])
plt.grid(alpha=0.3)
plt.show()

# plot 2
lamb = np.arange(0, 15, 0.01)

theta_record = []

for k in lamb:
    theta = cvx.Variable([10, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) + k*cvx.sum_squares(theta))
    prob = cvx.Problem(obj)
    result = prob.solve()
    theta_record.append(np.ravel(theta.value))

plt.figure(figsize=(10, 8))
plt.plot(lamb, theta_record, linewidth=1)
plt.title('ridge coefficients as a function of regularization', fontsize=15)
plt.xlabel('$\lambda$', fontsize=15)
plt.ylabel(r'weight $\theta$', fontsize=15)
plt.show()
# ----------------------------------------------------------------------------------------------------
# LASSO


def LASSO(d, m, lamb):

    u = np.linspace(np.min(x), np.max(x), d)

    A = np.hstack([np.exp(-(x-u[i])**2/(2*m**2)) for i in range(d)])
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*m**2))
                         for i in range(d)])

    A = np.asmatrix(A)
    rbfbasis = np.asmatrix(rbfbasis)

    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) +
                       lamb*cvx.norm(theta, 1))
    prob = cvx.Problem(obj)
    result = prob.solve()

    return rbfbasis, theta


rbfbasis, theta = LASSO(15, 1, 10)
yp = rbfbasis*theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', color='k', label='data')
plt.plot(xp, yp, label='LASSO')
plt.title('LASSO regularization(L1)', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# plot(LASSO)

# plot 1
plt.figure(figsize=(10, 8))
plt.title(r'LASSO: magnitude of $\theta$', fontsize=15)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel('magnitude', fontsize=15)
plt.stem(np.linspace(1, 15, 15).reshape(-1, 1), theta.value)
plt.xlim([0, 16])
plt.ylim([np.min(theta.value), np.max(theta.value)])
plt.grid(alpha=0.3)
plt.show()

# plot 2
lamb = np.arange(0, 15, 0.01)

theta_record = []

for k in lamb:
    theta = cvx.Variable([10, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) + k*cvx.norm(theta, 1))
    prob = cvx.Problem(obj)
    result = prob.solve()
    theta_record.append(np.ravel(theta.value))

plt.figure(figsize=(10, 8))
plt.plot(lamb, theta_record, linewidth=1)
plt.title('LASSO coefficients as a function of regularization', fontsize=15)
plt.xlabel('$\lambda$', fontsize=15)
plt.ylabel(r'weight $\theta$', fontsize=15)
plt.show()
# ----------------------------------------------------------------------------------------------------
# reduced order model

# L2 - penalty


def L2_penalty(d, m, lamb):

    u = np.linspace(np.min(x), np.max(x), d)

    A = np.hstack([np.exp(-(x-u[i])**2/(2*m**2)) for i in range(d)])
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*m**2))
                         for i in range(d)])

    A = np.asmatrix(A)
    rbfbasis = np.asmatrix(rbfbasis)

    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) +
                       lamb*cvx.norm(theta, 2))
    prob = cvx.Problem(obj)
    result = prob.solve()

    return rbfbasis, theta


rbfbasis, theta = L2_penalty(15, 1, 10)
yp = rbfbasis*theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', color='k', label='data')
plt.plot(xp, yp, label='L2 penalty')
plt.title('L2 penalty', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# L2 & L1 penalty

# L2 penalty
x = np.arange(-4, 4, 0.01)
k = 4
y = (x-1)**2 + 1/6*(x-1)**3 + k*x**2

x_star = x[np.argmin(y)]
print(x_star)

plt.plot(x, y, 'g', linewidth=2.5)
plt.axvline(x=x_star, color='k', linewidth=1, linestyle='--')
plt.ylim([np.min(y), np.max(y)])
plt.show()

for k in [0, 1, 2, 4]:
    y = (x-1)**2 + 1/6*(x-1)**3 + k*x**2
    x_star = x[np.argmin(y)]

    plt.plot(x, y, 'g', linewidth=2.5)
    plt.axvline(x=x_star, color='k', linewidth=1, linestyle='--')
    plt.ylim([0, 10])
    plt.title('ridge: k = {}'.format(k))
    plt.show()

# L1 penalty
x = np.arange(-4, 4, 0.01)
k = 2
y = (x-1)**2 + 1/6*(x-1)**3 + k*abs(x)

x_star = x[np.argmin(y)]
print(x_star)

plt.plot(x, y, 'g', linewidth=2.5)
plt.axvline(x=x_star, color='k', linewidth=1, linestyle='--')
plt.ylim([0, 10])
plt.show()

for k in [0, 1, 2]:
    y = (x-1)**2 + 1/6*(x-1)**3 + k*abs(x)
    x_star = x[np.argmin(y)]

    plt.plot(x, y, 'g', linewidth=2.5)
    plt.axvline(x=x_star, color='k', linewidth=1, linestyle='--')
    plt.ylim([0, 10])
    plt.title('LASSO: k = {}'.format(k))
    plt.show()
# ====================================================================================================
