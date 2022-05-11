# ====================================================================================================
# linear algebra
# ====================================================================================================
# linear equations

import numpy as np

# example 1
A = np.array([[4, -5],
              [-2, 3]])

b = np.array([[-13],
              [9]])  # column vec / row vec: np.array([-13, 9])

x = np.linalg.inv(A).dot(b)
print(x)

A = np.asmatrix(A)
b = np.asmatrix(b)

x = A.I*b
print(x)

# example 2
x = np.array([[1],
              [1]])

y = np.array([[2],
              [3]])

print(x.T.dot(y))

x = np.asmatrix(x)
y = np.asmatrix(y)

print(x.T*y)

z = x.T*y
print(z.A)
# ----------------------------------------------------------------------------------------------------
# norms: strength or distance in linear space

# l2 & l1 norm
x = np.array([[4],
              [3]])

np.linalg.norm(x, 2)
np.linalg.norm(x)  # default: l2 norm
np.linalg.norm(x, 1)

# orthogonality
x = np.matrix([[1], [2]])
y = np.matrix([[2], [-1]])

x.T*y  # orthogonal
# ----------------------------------------------------------------------------------------------------
# matrix and linear transformation

# rotation
theta = 90/180*np.pi  # radian value
R = np.matrix([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]])
x = np.matrix([[1], [0]])

y = R*x
print(y)

# stretch & compress

# projection
P = np.matrix([[1, 0],
               [0, 0]])
x = np.matrix([[1], [1]])

y = P*x
print(y)

# linear transformation
A = np.array([[1, 0],
              [0, 0]])
D, V = np.linalg.eig(A)

print('D :', D)  # eigenvalue
print('V :', V)  # eigenvector(column)
# ----------------------------------------------------------------------------------------------------
# system of linear equations(AX=B)

# # of unknown & equations(constraints): =, >, <

# A: fat or skinny

# (1) well-determined:(=)
# (2) under-determined(>, fat)
# (3) over-determined(<, skinny)
# ----------------------------------------------------------------------------------------------------
# optimization point of view

# projection
X = np.matrix([[1], [1]])
Y = np.matrix([[2], [0]])

print(X)
print(Y)
print(X.T*Y)
print(Y.T*Y)

omega = (X.T*Y)/(Y.T*Y)
print(float(omega))

omega = float(omega) # omega: matrix -> float
W = omega*Y
print(W)

A = np.matrix([[1, 0], [0, 1], [0, 0]])
B = np.matrix([[1], [1], [1]])

X = (A.T*A).I*A.T*B
print(X)

Bstar = A*X
print(Bstar)
# ====================================================================================================
