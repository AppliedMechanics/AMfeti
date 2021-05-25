import numpy as np
from scipy.sparse.linalg import spsolve as spp
from matplotlib import pyplot as plt


def solve(A_operator, b_operator, lambda_init):
    rk = b_operator - np.dot(A_operator, lambda_init)

    MaxIter = A_operator.shape[1]
    P = dict()
    Q = dict()
    Store = []
    Store_residual = []

    for k in range(MaxIter):
        pk = rk
        for i in range(k):
            Orthoparameter = np.dot(np.conjugate(Q[i]).T, pk) / np.dot(np.conjugate(Q[i]).T, P[i])
            pk =  pk  -Orthoparameter * P[i]

        P[k] = pk
        Q[k] = np.dot(A_operator, pk)

        if k > 0:
            Store.append(np.dot(P[k-1].T, np.dot(A_operator, P[k])))

        alpha_numerator = np.dot(np.conjugate(P[k]).T, rk)
        alpha_denominator = np.dot(np.conjugate(P[k]).T, Q[k])

        alpha_k = (alpha_numerator / alpha_denominator)

        lambda_init = lambda_init + P[k] * alpha_k



        #Current residual
        rk = rk - alpha_k * Q[k]



        if k > 0:
            Store_residual.append(np.linalg.norm(rk))

        print('CG residual', np.linalg.norm(rk))
        print('Actual reisdual', np.linalg.norm(b - np.dot(A_operator, lambda_init)))

        if np.linalg.norm(rk) < 1e-5:
            break
        # print('The current norm is', np.linalg.norm(rk/b))

    return lambda_init, k, Store, Store_residual


SelectSize = 100
A = np.random.rand(SelectSize, SelectSize)
Ai = np.random.rand(SelectSize, SelectSize)
b = np.random.rand(SelectSize)
lambda_init = np.zeros(SelectSize)


A = A
A = A + np.conjugate(A).T

FinalSolutionIterative, iteration_count, Store, Store_residual = solve(A, b, lambda_init)

FinalSolutionDirect = spp(A, b)

plt.figure()
plt.plot(np.arange(iteration_count), np.abs(Store))
# plt.plot(np.arange(iteration_count), np.abs(Store_residual))
plt.yscale('log')
plt.show()

Error = np.linalg.norm(FinalSolutionDirect - FinalSolutionIterative) / np.linalg.norm(FinalSolutionDirect)

print('The error in relative terms is ', Error)
print('Iterations required for convergence', iteration_count)