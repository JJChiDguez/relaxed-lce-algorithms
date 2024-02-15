#!/usr/bin/sage -python
# -*- coding: utf8 -*-

# ++++++++++
# IMPORTANT: This example is expected to run in polynomial time but not necessary for q smaller than n
# ++++++++++

import sys

from lib import (
    arguments,
    sample_permutation_matrix,
    oracle_call_ipce,
    algorithm,
)

# SageMath imports
from sage.all import (
    identity_matrix,
    FiniteField,
)

def main(n, k, q, Parallel=False, bench=False):

    def setup():
        P = sample_permutation_matrix(n, q)
        print(f'\nSecret permutation matrix, Q:\n{P}\n')
        def IPCE_oracle():
            return oracle_call_ipce(P, n, k, q)
        return IPCE_oracle

    IPCE_oracle = setup()
    M, M_, _M = IPCE_oracle()

    G = identity_matrix(FiniteField(q), k).augment(M, subdivide=False)
	# Recover permutation matrix
    P_ = algorithm(k, n, q, M, M_, _M, M, Parallel=Parallel, bench=bench)
    if P_ is None:
        return False
    print(f'Recovered permutation matrix, Q\':\n{P_}')
	
    return (G * P_).rref()[:k,k:n] == M_ and (G * (P_.transpose())).rref()[:k,k:n] == _M

if __name__ == '__main__':

    n = arguments(sys.argv[1:]).code_size_n
    assert(not n % 2)
    k = n // 2
    q = arguments(sys.argv[1:]).prime
    bench = arguments(sys.argv[1:]).benchmark

    print(f'\ncode dimension, k:\t{k}')
    print(f'code length, n:\t\t{n}')
    print(f'Field size, q:\t\t{q}\n')

    n_tests = {True:10, False:1}[bench]
    success = 0
    failure = 0

    print(f'Running {n_tests} tests')
    for x in range(n_tests):
        if main(n, k, q, Parallel=True, bench=bench):
            success = success + 1
        else:
            failure = failure + 1
    print(f'\nSuccess: {success}, Failure: {failure}')