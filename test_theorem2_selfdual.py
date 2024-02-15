#!/usr/bin/sage -python
# -*- coding: utf8 -*-

# ++++++++++
# IMPORTANT: This example is expected to run in polynomial time but not necessary for too much small values of q and n
# ++++++++++

from lib import (
    sample_monomial_matrix,
	algorithm,
)

# SageMath imports
from sage.all import (
    matrix,
    identity_matrix,
    FiniteField,
)


def main(code, q, Parallel=False):

    k, n = code.dimensions()
    print(f'\ncode dimension, k:\t{k}')
    print(f'code length, n:\t\t{n}')
    print(f'Field size, q:\t\t{q}\n')
    print(f'\nOriginal code:\n{code}\n')

    def setup():
        R = sample_monomial_matrix(n, q)
        A_code = (code * R).rref()
        while A_code[0:k,0:k] != identity_matrix(FiniteField(q), k, k):
            R = sample_monomial_matrix(n, q)
            A_code = (code * R).rref()

        M = sample_monomial_matrix(n, q)
        B_code = (A_code * M).rref()
        C_code = (A_code * (M.inverse())).rref()
        while B_code[0:k,0:k] != identity_matrix(FiniteField(q), k, k) or C_code[0:k,0:k] != identity_matrix(FiniteField(q), k, k) or B_code == C_code:
            M = sample_monomial_matrix(n, q)
            B_code = (A_code * M).rref()
            C_code = (A_code * (M.inverse())).rref()

        print(f'\nSecret monomial matrix, Q:\n{M}\n')
        return A_code[0:k,k:n], B_code[0:k,k:n], C_code[0:k,k:n]

    M, M_, _M = setup()

    G = identity_matrix(FiniteField(q), k).augment(M, subdivide=False)
	# Recover monomial matrix
    Q_ = algorithm(k, n, q, M, M_, _M, M, Parallel=Parallel)
    if Q_ is None:
        return False
    print(f'Recovered monomial matrix, Q\':\n{Q_}')
	
    return (G * Q_).rref()[:k,k:n] == M_ and (G * (Q_.inverse())).rref()[:k,k:n] == _M

if __name__ == '__main__':

    q = 7
    code7_4 = matrix(FiniteField(q), 2, 4, [
    6,5,1,1,
    1,6,5,1
    ])

    code7_12 = matrix(FiniteField(q), 6, 12, [
    2,6,1,0,0,0,0,1,1,5,3,0,
    0,2,5,0,0,0,0,1,1,1,6,3,
    1,5,3,0,2,6,1,0,0,0,0,1,
    1,1,6,3,0,2,5,0,0,0,0,1,
    0,0,0,1,1,5,3,0,2,6,1,0,
    0,0,0,1,1,1,6,3,0,2,5,0
    ])

    code7_16 = matrix(FiniteField(q), 8, 16, [
    2,6,1,0,0,0,0,0,0,0,0,1,1,5,3,0,
    0,2,5,0,0,0,0,0,0,0,0,1,1,1,6,3,
    1,5,3,0,2,6,1,0,0,0,0,0,0,0,0,1,
    1,1,6,3,0,2,5,0,0,0,0,0,0,0,0,1,
    0,0,0,1,1,5,3,0,2,6,1,0,0,0,0,0,
    0,0,0,1,1,1,6,3,0,2,5,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,5,3,0,2,6,1,0,
    0,0,0,0,0,0,0,1,1,1,6,3,0,2,5,0
    ])

    code7_24 = matrix(FiniteField(q), 12, 24, [
    4,5,5,2,3,2,5,0,1,3,1,4,4,0,0,1,1,4,2,1,0,5,2,0,
    6,4,5,1,3,0,2,3,6,4,0,2,2,4,0,6,3,5,0,4,2,3,5,6,
    0,5,2,0,4,5,5,2,3,2,5,0,1,3,1,4,4,0,0,1,1,4,2,1,
    2,3,5,6,6,4,5,1,3,0,2,3,6,4,0,2,2,4,0,6,3,5,0,4,
    1,4,2,1,0,5,2,0,4,5,5,2,3,2,5,0,1,3,1,4,4,0,0,1,
    3,5,0,4,2,3,5,6,6,4,5,1,3,0,2,3,6,4,0,2,2,4,0,6,
    4,0,0,1,1,4,2,1,0,5,2,0,4,5,5,2,3,2,5,0,1,3,1,4,
    2,4,0,6,3,5,0,4,2,3,5,6,6,4,5,1,3,0,2,3,6,4,0,2,
    1,3,1,4,4,0,0,1,1,4,2,1,0,5,2,0,4,5,5,2,3,2,5,0,
    6,4,0,2,2,4,0,6,3,5,0,4,2,3,5,6,6,4,5,1,3,0,2,3,
    3,2,5,0,1,3,1,4,4,0,0,1,1,4,2,1,0,5,2,0,4,5,5,2,
    3,0,2,3,6,4,0,2,2,4,0,6,3,5,0,4,2,3,5,6,6,4,5,1
    ])

    code7_28 = matrix(FiniteField(q), 14, 28, [
    3,6,1,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,
    5,2,5,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,
    3,4,3,6,3,6,1,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,
    0,5,3,5,5,2,5,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,
    4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,
    3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,
    5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,1,1,6,4,3,4,1,2,3,
    3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,1,1,2,0,2,4,2,2,5,
    4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,1,1,6,4,3,
    4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,1,1,2,0,2,
    1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,1,
    1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,1,
    0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,
    0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0
    ])

    code7_36 = matrix(FiniteField(q), 18, 36, [
    1,1,3,1,6,1,6,5,1,4,4,2,4,0,0,1,5,2,6,3,1,5,0,4,0,0,1,3,4,1,1,4,1,0,4,4,
    5,0,4,0,0,3,6,3,3,5,3,3,5,2,1,0,0,5,3,3,3,4,0,2,2,2,5,4,5,1,1,1,5,1,6,2,
    1,0,4,4,1,1,3,1,6,1,6,5,1,4,4,2,4,0,0,1,5,2,6,3,1,5,0,4,0,0,1,3,4,1,1,4,
    5,1,6,2,5,0,4,0,0,3,6,3,3,5,3,3,5,2,1,0,0,5,3,3,3,4,0,2,2,2,5,4,5,1,1,1,
    4,1,1,4,1,0,4,4,1,1,3,1,6,1,6,5,1,4,4,2,4,0,0,1,5,2,6,3,1,5,0,4,0,0,1,3,
    5,1,1,1,5,1,6,2,5,0,4,0,0,3,6,3,3,5,3,3,5,2,1,0,0,5,3,3,3,4,0,2,2,2,5,4,
    0,0,1,3,4,1,1,4,1,0,4,4,1,1,3,1,6,1,6,5,1,4,4,2,4,0,0,1,5,2,6,3,1,5,0,4,
    2,2,5,4,5,1,1,1,5,1,6,2,5,0,4,0,0,3,6,3,3,5,3,3,5,2,1,0,0,5,3,3,3,4,0,2,
    1,5,0,4,0,0,1,3,4,1,1,4,1,0,4,4,1,1,3,1,6,1,6,5,1,4,4,2,4,0,0,1,5,2,6,3,
    3,4,0,2,2,2,5,4,5,1,1,1,5,1,6,2,5,0,4,0,0,3,6,3,3,5,3,3,5,2,1,0,0,5,3,3,
    5,2,6,3,1,5,0,4,0,0,1,3,4,1,1,4,1,0,4,4,1,1,3,1,6,1,6,5,1,4,4,2,4,0,0,1,
    0,5,3,3,3,4,0,2,2,2,5,4,5,1,1,1,5,1,6,2,5,0,4,0,0,3,6,3,3,5,3,3,5,2,1,0,
    4,0,0,1,5,2,6,3,1,5,0,4,0,0,1,3,4,1,1,4,1,0,4,4,1,1,3,1,6,1,6,5,1,4,4,2,
    5,2,1,0,0,5,3,3,3,4,0,2,2,2,5,4,5,1,1,1,5,1,6,2,5,0,4,0,0,3,6,3,3,5,3,3,
    1,4,4,2,4,0,0,1,5,2,6,3,1,5,0,4,0,0,1,3,4,1,1,4,1,0,4,4,1,1,3,1,6,1,6,5,
    3,5,3,3,5,2,1,0,0,5,3,3,3,4,0,2,2,2,5,4,5,1,1,1,5,1,6,2,5,0,4,0,0,3,6,3,
    6,1,6,5,1,4,4,2,4,0,0,1,5,2,6,3,1,5,0,4,0,0,1,3,4,1,1,4,1,0,4,4,1,1,3,1,
    0,3,6,3,3,5,3,3,5,2,1,0,0,5,3,3,3,4,0,2,2,2,5,4,5,1,1,1,5,1,6,2,5,0,4,0
    ])

    code7_40 = matrix(FiniteField(q), 20, 40, [
    6,0,1,3,0,2,3,1,4,5,1,5,2,0,6,3,6,0,1,3,0,0,1,6,5,4,0,3,3,4,4,0,6,6,3,0,2,2,1,1,
    4,5,3,1,0,3,5,6,5,0,2,4,1,2,4,5,6,3,3,2,6,2,2,0,0,6,1,2,5,5,6,5,4,5,0,5,2,3,5,5,
    2,2,1,1,6,0,1,3,0,2,3,1,4,5,1,5,2,0,6,3,6,0,1,3,0,0,1,6,5,4,0,3,3,4,4,0,6,6,3,0,
    2,3,5,5,4,5,3,1,0,3,5,6,5,0,2,4,1,2,4,5,6,3,3,2,6,2,2,0,0,6,1,2,5,5,6,5,4,5,0,5,
    6,6,3,0,2,2,1,1,6,0,1,3,0,2,3,1,4,5,1,5,2,0,6,3,6,0,1,3,0,0,1,6,5,4,0,3,3,4,4,0,
    4,5,0,5,2,3,5,5,4,5,3,1,0,3,5,6,5,0,2,4,1,2,4,5,6,3,3,2,6,2,2,0,0,6,1,2,5,5,6,5,
    3,4,4,0,6,6,3,0,2,2,1,1,6,0,1,3,0,2,3,1,4,5,1,5,2,0,6,3,6,0,1,3,0,0,1,6,5,4,0,3,
    5,5,6,5,4,5,0,5,2,3,5,5,4,5,3,1,0,3,5,6,5,0,2,4,1,2,4,5,6,3,3,2,6,2,2,0,0,6,1,2,
    5,4,0,3,3,4,4,0,6,6,3,0,2,2,1,1,6,0,1,3,0,2,3,1,4,5,1,5,2,0,6,3,6,0,1,3,0,0,1,6,
    0,6,1,2,5,5,6,5,4,5,0,5,2,3,5,5,4,5,3,1,0,3,5,6,5,0,2,4,1,2,4,5,6,3,3,2,6,2,2,0,
    0,0,1,6,5,4,0,3,3,4,4,0,6,6,3,0,2,2,1,1,6,0,1,3,0,2,3,1,4,5,1,5,2,0,6,3,6,0,1,3,
    6,2,2,0,0,6,1,2,5,5,6,5,4,5,0,5,2,3,5,5,4,5,3,1,0,3,5,6,5,0,2,4,1,2,4,5,6,3,3,2,
    6,0,1,3,0,0,1,6,5,4,0,3,3,4,4,0,6,6,3,0,2,2,1,1,6,0,1,3,0,2,3,1,4,5,1,5,2,0,6,3,
    6,3,3,2,6,2,2,0,0,6,1,2,5,5,6,5,4,5,0,5,2,3,5,5,4,5,3,1,0,3,5,6,5,0,2,4,1,2,4,5,
    2,0,6,3,6,0,1,3,0,0,1,6,5,4,0,3,3,4,4,0,6,6,3,0,2,2,1,1,6,0,1,3,0,2,3,1,4,5,1,5,
    1,2,4,5,6,3,3,2,6,2,2,0,0,6,1,2,5,5,6,5,4,5,0,5,2,3,5,5,4,5,3,1,0,3,5,6,5,0,2,4,
    4,5,1,5,2,0,6,3,6,0,1,3,0,0,1,6,5,4,0,3,3,4,4,0,6,6,3,0,2,2,1,1,6,0,1,3,0,2,3,1,
    5,0,2,4,1,2,4,5,6,3,3,2,6,2,2,0,0,6,1,2,5,5,6,5,4,5,0,5,2,3,5,5,4,5,3,1,0,3,5,6,
    0,2,3,1,4,5,1,5,2,0,6,3,6,0,1,3,0,0,1,6,5,4,0,3,3,4,4,0,6,6,3,0,2,2,1,1,6,0,1,3,
    0,3,5,6,5,0,2,4,1,2,4,5,6,3,3,2,6,2,2,0,0,6,1,2,5,5,6,5,4,5,0,5,2,3,5,5,4,5,3,1
    ])

    code7_44 = matrix(FiniteField(q), 22, 44, [
    0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,
    2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,
    6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,
    6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,
    4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,
    0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,
    4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,
    0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,
    3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,
    2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,
    2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,
    6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,
    0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,
    0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,
    4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,6,5,4,5,
    3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,2,6,5,0,
    6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,6,5,5,0,
    2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,2,4,1,2,
    6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,6,3,3,0,
    2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2,3,2,5,1,
    6,3,3,0,6,5,5,0,6,5,4,5,4,6,1,0,0,6,6,0,2,0,1,2,3,2,0,4,4,3,3,5,4,6,3,0,6,2,2,0,0,6,0,2,
    3,2,5,1,2,4,1,2,2,6,5,0,3,2,5,0,0,6,6,4,6,5,5,1,2,0,4,5,0,2,5,6,0,6,0,1,6,3,4,6,2,5,5,2
    ])

    code7_52 = matrix(FiniteField(q), 26, 52, [
    3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,
    5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,
    3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,
    0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,
    4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,
    3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,
    5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,
    3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,
    4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,
    4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,
    1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,6,4,3,4,1,2,3,5,4,0,3,4,1,5,0,3,4,3,6,3,6,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,0,2,4,2,2,5,3,6,2,0,3,3,3,2,0,5,3,5,5,2,5,0
    ])

    code7_56 = matrix(FiniteField(q), 28, 56, [
    5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,
    0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,
    2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,
    1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,
    3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,
    2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,
    6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,
    1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,
    3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,
    6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,
    0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,5,3,0,3,5,6,4,2,2,3,0,5,3,2,5,4,0,5,1,6,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,6,3,0,2,1,4,2,4,2,5,0,3,1,1,1,4,0,5,2,0
    ])

    n_tests = 25

    for code in (code7_16,code7_24,code7_28):
        success = 0
        failure = 0
        print(f'Running {n_tests} tests')
        for x in range(n_tests):
            if main(code, q, Parallel=True):
                success = success + 1
            else:
                failure = failure + 1
        print(f'\nSuccess: {success}, Failure: {failure}')
