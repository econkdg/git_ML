# ====================================================================================================
# regularization_scikit learn
# ====================================================================================================
# model: y = theta0 + theta1*x + theta2*x^2 + theta3*x^3 + noise
# ----------------------------------------------------------------------------------------------------
# import library

from sklearn import linear_model
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats
import FinanceDataReader as fdr
import os
# ----------------------------------------------------------------------------------------------------
# generate random data

n = 100
x = -5 + 10*np.random.rand(n, 1)  # uniform: [-5, 10]
noise = 5*np.random.randn(n, 1)  # Gaussian

y = 10 + 0.70*x + 0.35*(x-5)**2 + 0.25*x**3 + noise

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)
# ----------------------------------------------------------------------------------------------------
# scikit ridge


def scikit_ridge(d, sigma):

    u = np.linspace(np.min(x), np.max(x), d)

    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2))
                   for i in range(d)])
    Ap = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2))
                    for i in range(d)])

    reg = linear_model.Ridge(alpha=10)
    fit = reg.fit(A, y)
    predict_ridge = reg.predict(Ap)

    return predict_ridge


predict_ridge = scikit_ridge(15, 1)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', color='k', label='data')
plt.plot(xp, predict_ridge, label='ridge')
plt.title('ridge regression (L2)', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# scikit LASSO


def scikit_LASSO(d, sigma):

    u = np.linspace(np.min(x), np.max(x), d)

    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2))
                   for i in range(d)])
    Ap = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2))
                    for i in range(d)])

    reg = linear_model.Lasso(alpha=10)
    fit = reg.fit(A, y)
    predict_LASSO = reg.predict(Ap)

    return predict_LASSO


predict_LASSO = scikit_LASSO(15, 1)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', color='k', label='data')
plt.plot(xp, predict_LASSO, label='LASSO')
plt.title('LASSO regression (L1)', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
# ----------------------------------------------------------------------------------------------------
# class scikit regularization

# generate random data
n = 100
x = -5 + 10*np.random.rand(n, 1)  # uniform: [-5, 10]
noise = 5*np.random.randn(n, 1)  # Gaussian

y = 10 + 0.70*x + 0.35*(x-5)**2 + 0.25*x**3 + noise

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

# read time series data(DataFrame)


class cleansing:

    def __init__(self, code, start, end, t):

        self.code = code
        self.start = start
        self.end = end
        self.t = t

    def gen_ln_return(self):

        raw = fdr.DataReader(self.code, self.start, self.end)

        raw['ln_Close'] = np.log(raw['Close'])
        raw['lag_ln_Close'] = raw['ln_Close'].shift(self.t)

        raw['return'] = raw['ln_Close'] - raw['lag_ln_Close']

        ln_return = raw['return']
        ln_return = pd.DataFrame(ln_return).reset_index()

        return ln_return


data = [cleansing('TSLA', '2018-01-01', '2020-12-31', 1),
        cleansing('AAPL', '2018-01-01', '2020-12-31', 1),
        cleansing('MSFT', '2018-01-01', '2020-12-31', 1)]

Y = data[0].gen_ln_return()
Y.rename(columns={'return': 'Y_return'}, inplace=True)

X = data[1].gen_ln_return()
X.rename(columns={'return': 'X_return'}, inplace=True)

df = pd.merge(Y, X)
df = df.dropna(axis=0)

df.isnull().sum()


def date_function(x):

    # numerical type, not string type -> indexing is not applied
    result = str(x)

    return result[0:4] + '-' + result[5:7] + '-' + result[8:10]


df['date'] = pd.DataFrame(df['Date'].apply(date_function))

df.drop(['Date'], axis=1, inplace=True)

y = np.array(df['Y_return']).reshape(-1, 1)
x = np.array(df['X_return']).reshape(-1, 1)

xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

# ridge & LASSO


class regularization:

    def __init__(self, d, sigma, alpha, L_n):

        self.d = d
        self.sigma = sigma
        self.alpha = alpha
        self.L_n = L_n

    def L_n_penalty(self):

        u = np.linspace(np.min(x), np.max(x), self.d)

        A = np.hstack([np.exp(-(x-u[i])**2/(2*self.sigma**2))
                      for i in range(self.d)])
        rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*self.sigma**2))
                             for i in range(self.d)])

        A = np.asmatrix(A)
        rbfbasis = np.asmatrix(rbfbasis)

        theta = cvx.Variable([self.d, 1])
        obj = cvx.Minimize(cvx.sum_squares(A*theta - y) +
                           self.alpha*cvx.norm(theta, self.L_n))
        prob = cvx.Problem(obj)
        result = prob.solve()

        return theta.value

    def scikit_ridge(self):

        u = np.linspace(np.min(x), np.max(x), self.d)

        A = np.hstack([np.exp(-(x-u[i])**2/(2*self.sigma**2))
                      for i in range(self.d)])
        Ap = np.hstack([np.exp(-(xp-u[i])**2/(2*self.sigma**2))
                       for i in range(self.d)])

        reg = linear_model.Ridge(alpha=self.alpha)
        fit = reg.fit(A, y)

        predict_ridge = reg.predict(Ap)

        return predict_ridge

    def plot_ridge(self):

        plt.figure(figsize=(10, 8))
        plt.plot(x, y, 'o', color='k', label='data')
        plt.plot(xp, self.scikit_ridge(), label='ridge')
        plt.title('ridge regression (L2)', fontsize=15)
        plt.xlabel('X', fontsize=15)
        plt.ylabel('Y', fontsize=15)
        plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
        plt.legend(fontsize=15)
        plt.grid(alpha=0.3)

        fig_ridge = plt.show()

        return fig_ridge

    def scikit_LASSO(self):

        u = np.linspace(np.min(x), np.max(x), self.d)

        A = np.hstack([np.exp(-(x-u[i])**2/(2*self.sigma**2))
                      for i in range(self.d)])
        Ap = np.hstack([np.exp(-(xp-u[i])**2/(2*self.sigma**2))
                       for i in range(self.d)])

        reg = linear_model.Lasso(alpha=self.alpha)
        fit = reg.fit(A, y)

        predict_LASSO = reg.predict(Ap)

        return predict_LASSO

    def plot_LASSO(self):

        plt.figure(figsize=(10, 8))
        plt.plot(x, y, 'o', color='k', label='data')
        plt.plot(xp, self.scikit_LASSO(), label='LASSO')
        plt.title('LASSO regression (L1)', fontsize=15)
        plt.xlabel('X', fontsize=15)
        plt.ylabel('Y', fontsize=15)
        plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
        plt.legend(fontsize=15)
        plt.grid(alpha=0.3)

        fig_LASSO = plt.show()

        return fig_LASSO


model = [regularization(20, 1, 0.005, 2),
         regularization(20, 1, 0.001, 1),
         regularization(15, 1, 0.005, 2),
         regularization(15, 1, 0.001, 1),
         regularization(10, 1, 0.005, 2),
         regularization(10, 1, 0.001, 1),
         regularization(5, 1, 0.005, 2),
         regularization(5, 1, 0.001, 1)]
# ====================================================================================================
