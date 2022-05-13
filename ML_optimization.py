# ====================================================================================================
# optimization
# ====================================================================================================
# CVXPY(convex problem)

import numpy as np
import cvxpy as cvx

# example 1
f = np.array([[3], [3/2]])
lb = np.array([[-1], [0]])  # lower bound
ub = np.array([[2], [3]])  # upper bound

x = cvx.Variable([2, 1])  # decision variable

obj = cvx.Minimize(-f.T*x)
constraints = [lb <= x, x <= ub]  # cvxpy grammer -> comparing elementwise

# prob = cvx.Problem(obj) or prob = cvx.Problem(obj, [])
prob = cvx.Problem(obj, constraints)
result = prob.solve()

print(x.value)  # optimal solution
print(result)  # objective function value

# example 2(quadratic programming)
f = np.array([[3], [4]])
H = np.array([[1/2, 0], [0, 0]])

A = np.array([[-1, -3], [2, 5], [3, 4]])
b = np.array([[-15], [100], [80]])
lb = np.array([[0], [0]])

x = cvx.Variable([2, 1])

obj = cvx.Minimize(cvx.quad_form(x, H) + f.T*x)  # cvx.quad_form?
constraints = [A*x <= b, lb <= x]

prob = cvx.Problem(obj, constraints)
result = prob.solve()

print(x.value)
print(result)

# example 3(shortest distance)
f = np.array([[-6], [-6]])
H = np.array([[1, 0], [0, 1]])

A = np.array([[1, 1]])
b = 3
lb = np.array([[0], [0]])

x = cvx.Variable([2, 1])

obj = cvx.Minimize(cvx.quad_form(x, H) + f.T*x)
constraints = [A*x <= b, lb <= x]

prob = cvx.Problem(obj, constraints)
result = prob.solve()

print(x.value)
print(result)

# example 4(shortest distance)


def optimizer(x1, y1, x2, y2):
    a = np.array([[x1], [y1]])
    b = np.array([[x2], [y2]])

    Aeq = np.array([[0, 1]])
    beq = 0

    x = cvx.Variable([2, 1])

    mu = 1
    obj = cvx.Minimize(cvx.norm(a-x, 2) + mu*cvx.norm(b-x, 2))
    constraints = [Aeq*x == beq]  # trick

    prob = cvx.Problem(obj, constraints)
    result = prob.solve()

    return x.value, result


print(optimizer(0, 1, 4, 2))
print(np.sqrt(4**2 + 3**2))

# example 5(logistics)


def logistics(x1, y1, x2, y2, x3, y3):
    a = np.array([[np.sqrt(x1)], [y1]])
    b = np.array([[-np.sqrt(x2)], [y2]])
    c = np.array([[x3], [y3]])

    x = cvx.Variable([2, 1])

    obj = cvx.Minimize(cvx.norm(a-x, 2) + cvx.norm(b-x, 2) + cvx.norm(c-x, 2))
    #obj = cvx.Minimize(cvx.norm(a-x, 1) + cvx.norm(b-x, 1) + cvx.norm(c-x, 1))

    prob = cvx.Problem(obj)
    result = prob.solve()

    return x.value, result


print(logistics(3, 0, 3, 0, 0, 3))

b1 = np.array([[np.sqrt(3)], [0]])
b2 = np.array([[-np.sqrt(3)], [0]])
b3 = np.array([[0], [3]])

B = np.hstack([b1, b2, b3])
np.mean(B, 1)
# ----------------------------------------------------------------------------------------------------
# gradient descent


def gradient(n):
    H = np.matrix([[2, 0], [0, 2]])
    g = -np.matrix([[6], [6]])

    x = np.zeros((2, 1))
    alpha = 0.2

    for i in range(n):
        df = H*x + g
        x = x - alpha*df  # update rule

    return x


print(gradient(1000))
# ====================================================================================================
