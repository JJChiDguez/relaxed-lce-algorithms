#!/usr/bin/sage -python
# -*- coding: utf8 -*-

import sys

from lib import (
    arguments,
    sample_monomial_matrix,
    oracle_call_ilce,
    algorithm,
)

# SageMath imports
from sage.all import (
    identity_matrix,
    FiniteField,
)

import itertools
from timeit import default_timer as timer
import multiprocessing as mp

def main(n, k, q, Parallel=False, bench=False):
    def setup():
        Q = sample_monomial_matrix(n, q)
        print(f'\nSecret monomial matrix, Q:\n{Q}\n')
        def ILCE_oracle():
            return oracle_call_ilce(Q, n, k, q)
        return ILCE_oracle

    ILCE_oracle = setup()
    M, M_, _M = ILCE_oracle()

    G = identity_matrix(FiniteField(q), k).augment(M, subdivide=False)
	# Recover monomial matrix
    Q_ = algorithm(k, n, q, M, M_, _M, M, Parallel=Parallel, bench=bench)
    if Q_ is None:
        return False
    print(f'Recovered monomial matrix, Q\':\n{Q_}')
	
    return (G * Q_).rref()[:k,k:n] == M_ and (G * (Q_.inverse())).rref()[:k,k:n] == _M

if __name__ == '__main__':

	n = arguments(sys.argv[1:]).code_size_n
	assert(not n % 2)
	k = n // 2
	q = arguments(sys.argv[1:]).prime
	bench = arguments(sys.argv[1:]).benchmark

	print(f'\ncode dimension, k:\t{k}')
	print(f'code length, n:\t\t{n}')
	print(f'Field size, q:\t\t{q}\n')

	n_tests = {True:25, False:1}[bench]
	success = 0
	failure = 0

	print(f'Running {n_tests} tests')
	for x in range(n_tests):
		if main(n, k, q, Parallel=True, bench=bench):
			success = success + 1
		else:
			failure = failure + 1
	print(f'\nSuccess: {success}, Failure: {failure}')
