#!/usr/bin/sage -python
# -*- coding: utf8 -*-

import sys
from lib import (arguments,
    is_monomial,
    progress_bar,
	sample_monomial_matrix,
	oracle_call_ilce,
	get_linear_system,
)

# SageMath imports
from sage.all import (
    matrix,
    identity_matrix,
    FiniteField,
    ceil,
)

from timeit import default_timer as timer

def main(n, k, q):
    def setup():
        M = sample_monomial_matrix(n, q)
        print(f'\nSecret monomial matrix:\n{M}\n')
        def ILCE_oracle():
            return oracle_call_ilce(M, n, k, q)
        return ILCE_oracle

    ILCE_oracle = setup()

    preffix = identity_matrix(FiniteField(q), k)
    suffix  = identity_matrix(FiniteField(q), n - k)

    # Procedure starts below
    start = timer()

    expected_calls = ceil((n**2) / (2 * k * (n-k)))
    calls = 0
    linear_system = matrix(FiniteField(q), 0, n**2)
    for i in range(expected_calls):
        progress_bar(i, expected_calls, suffix='Building the system')
        A, B, C = ILCE_oracle()
        calls = calls + 1
        linear_system = linear_system.stack(get_linear_system(preffix, A, B, suffix))
        linear_system = linear_system.stack(get_linear_system(preffix, C, A, suffix))

    progress_bar(expected_calls, expected_calls, suffix='Building the system\n')
    
    middle = timer()
    kernel = linear_system.right_kernel().matrix()
    A_code = identity_matrix(FiniteField(q), k).augment(A, subdivide=False)
    for solution in kernel:
        M_ = matrix(n, n, solution)
        if is_monomial(M_, n):
            break
    end = timer() 

    print(f'\nRecovered monomial matrix:\n{M_}\n')
    print(f'\n# oracle calls:\t\t\t{calls}')
    print(f'Elapsed time (get linear sys):\t{middle - start} seconds')
    print(f'Elapsed time (recover matrix):\t{end - middle} seconds')
    print(f'Elapsed time (total):\t\t{end - start} seconds\n')
    return (A_code * M_).rref()[:k,k:n] == B and (A_code * (M_.inverse())).rref()[:k,k:n] == C

if __name__ == '__main__':

	k = arguments(sys.argv[1:]).code_dimension
	n = arguments(sys.argv[1:]).code_size_n
	q = arguments(sys.argv[1:]).prime
	bench = arguments(sys.argv[1:]).benchmark

	print(f'\ncode dimension, k:\t{k}')
	print(f'code length, n:\t\t{n}')
	print(f'Field size, q:\t\t{q}\n')

	n_tests = 1
	success = 0
	failure = 0

	print(f'Running {n_tests} tests')
	for x in range(n_tests):
		if main(n, k, q):
			success = success + 1
		else:
			failure = failure + 1
	print(f'\nSuccess: {success}, Failure: {failure}')
