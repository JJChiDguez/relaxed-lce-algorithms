#!/usr/bin/sage -python
# -*- coding: utf8 -*-

CYELLOW = '\33[33m'
CRED = '\033[91m'
CEND = '\033[0m'

import sys
import argparse
import csv
import tracemalloc

# SageMath imports
from sage.all import (
	randint,
    matrix,
    identity_matrix,
    random_matrix,
    zero_matrix,
    FiniteField,
    PolynomialRing,
)

from timeit import default_timer as timer
import multiprocessing as mp

# -------------------------------
def is_monomial(input_matrix, n):
	for i in range(0, n, 1):
		if len(input_matrix.nonzero_positions_in_row(i)) != 1:
			return False
	return True


# ----------------
def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        # Call the function being decorated and return the result
        return func(*args, **kwargs)
    wrapper.count = 0
    # Return the new decorated function
    return wrapper

# ----------------------------------------------
def progress_bar(count_value, total, suffix=''):
	bar_length = 100
	filled_up_Length = int(round(bar_length* count_value / float(total)))
	percentage = round(100.0 * count_value/float(total),1)
	bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
	sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
	sys.stdout.flush()


# -------------------------------
def arguments(args=sys.argv[1:]):
	parser = argparse.ArgumentParser(description="Parses command.")
	parser.add_argument("-k", "--code_dimension", type=int, help="code dimension", required=False)
	parser.add_argument("-m", "--code_size_m", type=int, help="code size (m)", required=False)
	parser.add_argument("-n", "--code_size_n", type=int, help="code size (n)", required=True)
	parser.add_argument("-q", "--prime", type=int, help="Field characteristic", required=True)
	parser.add_argument('-b', '--benchmark', action='store_true', help='Benchmark')

	if len(sys.argv) == 1:
		parser.print_help(sys.stderr)
		sys.exit(1)

	options = parser.parse_args(args)
	return options


# ------------------------------------------------------
def to_csv_file(file_name: str, data: list, label=True):
	with open(file_name, 'a', newline='') as file:
		writer = csv.writer(file)
		if label:
			field = ['n', 'q', '# vars of S_red', 'memory (gigabytes)', 'runtime (seconds)']
			writer.writerow(field)
		writer.writerow(data.values())


# ------------------------------------------------------
def to_csv_file_header(file_name: str, label=True):
	with open(file_name, 'a', newline='') as file:
		writer = csv.writer(file)
		if label:
			field = ['n', 'q', '# vars of S_red', 'memory (gigabytes)', 'runtime (seconds)']
			writer.writerow(field)
		

# ---------------------------------
def sample_invertible_matrix(k, q):
	while True:
		# Generate a random n x n matrix with entries in Zq
		A = random_matrix(FiniteField(q), k, k)
		# Check if the matrix is invertible
		if A.is_invertible():
			return A


# ----------------------------------------
def sample_secret_tensor_product(m, n, q):
	A = sample_invertible_matrix(m, q)
	B = sample_invertible_matrix(n, q)
	return (A.transpose()).tensor_product(B, subdivide=False)


# -------------------------------
def oracle_call_mce(M, mn, k, q):
	F = FiniteField(q)
	A_rand = random_matrix(F, k, mn - k)
	A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
	B_code = A_code * M
	W = B_code[0:k,0:k]
	while W.rank() < k :
		A_rand = random_matrix(F, k, mn - k)
		A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
		B_code = A_code * M
		W = B_code[0:k,0:k]
	FSB_code = (W.inverse())*B_code
	B = FSB_code[0:k,k:mn]
	return [A_rand,B]


# --------------------------------
def oracle_call_imce(M, mn, k, q):
	F = FiniteField(q)
	A_rand = random_matrix(F, k, mn - k)
	A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
	B_code = A_code * M
	W = B_code[0:k,0:k]
	C_code = A_code*(M.inverse())
	Z = C_code[0:k,0:k]
	while W.rank() < k or Z.rank() < k:
		A_rand = random_matrix(F, k, mn - k)
		A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
		B_code = A_code * M
		W = B_code[0:k,0:k]
		C_code = A_code*(M.inverse())
		Z = C_code[0:k,0:k]
	FSB_code = (W.inverse())*B_code
	B = FSB_code[0:k,k:mn]
	FSC_code = (Z.inverse())*C_code
	C = FSC_code[0:k,k:mn]
	return [A_rand,B,C]


# ----------------------------------
# sample permutation with fisher-yates
def sample_permutation_matrix(n, q):
	P = zero_matrix(FiniteField(q), n, n)
	a = [i for i in range(0, n)]
	for i in range(n-1, 0, -1):
		j = randint(0, i)
		tmp = a[i]
		a[i] = a[j]
		a[j] = tmp

	for i in range(0, n):
		P[i,a[i]] = 1

	return P


# -------------------------------
def sample_monomial_matrix(n, q):
    M = zero_matrix(FiniteField(q), n, n)
    for i in range(0, n):
       M[i,i] = randint(1, q-1)

    P = sample_permutation_matrix(n, q)
    return M * P


# ------------------------------
def oracle_call_lce(M, n, k, q):
	F = FiniteField(q)
	A_rand = random_matrix(F, k, n - k)
	A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
	B_code = A_code * M
	W = B_code[0:k,0:k]
	while W.rank() < k :
		A_rand = random_matrix(F, k, n - k)
		A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
		B_code = A_code * M
		W = B_code[0:k,0:k]
	FSB_code = (W.inverse())*B_code
	B = FSB_code[0:k,k:n]
	return [A_rand,B]


# -------------------------------
def oracle_call_ilce(M, n, k, q):
	F = FiniteField(q)
	A_rand = random_matrix(F, k, n - k)
	A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
	B_code = A_code * M
	W = B_code[0:k,0:k]
	C_code = A_code*(M.inverse())
	Z = C_code[0:k,0:k]
	while W.rank() < k or Z.rank() < k:
		A_rand = random_matrix(F, k, n - k)
		A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
		B_code = A_code * M
		W = B_code[0:k,0:k]
		C_code = A_code*(M.inverse())
		Z = C_code[0:k,0:k]
	
	FSB_code = (W.inverse())*B_code
	B = FSB_code[0:k,k:n]
	FSC_code = (Z.inverse())*C_code
	C = FSC_code[0:k,k:n]
	return [A_rand,B,C]


# ------------------------------
def oracle_call_pce(M, n, k, q):
	F = FiniteField(q)
	A_rand = random_matrix(F, k, n - k)
	A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
	B_code = A_code * M
	W = B_code[0:k,0:k]
	while W.rank() < k :
		A_rand = random_matrix(F, k, n - k)
		A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
		B_code = A_code * M
		W = B_code[0:k,0:k]
	FSB_code = (W.inverse())*B_code
	B = FSB_code[0:k,k:n]
	return [A_rand,B]


# -------------------------------
def oracle_call_ipce(M, n, k, q):
	F = FiniteField(q)
	A_rand = random_matrix(F, k, n - k)
	A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
	B_code = A_code * M
	W = B_code[0:k,0:k]
	C_code = A_code*(M.inverse())
	Z = C_code[0:k,0:k]
	while W.rank() < k or Z.rank() < k:
		A_rand = random_matrix(F, k, n - k)
		A_code = identity_matrix(F, k).augment(A_rand, subdivide=False)
		B_code = A_code * M
		W = B_code[0:k,0:k]
		C_code = A_code*(M.inverse())
		Z = C_code[0:k,0:k]
	FSB_code = (W.inverse())*B_code
	B = FSB_code[0:k,k:n]
	FSC_code = (Z.inverse())*C_code
	C = FSC_code[0:k,k:n]
	return [A_rand,B,C]


# ----------------------------------------------------
def get_monomial_matrix_from_linear_system(n, system):
	kernel = system.right_kernel().matrix()
	for solution in kernel:
		M_ = matrix(n, n, solution)
		if (M_.transpose() * M_).is_diagonal():
			return M_

	return None


# ------------------------------
def permutation_equations(n, q):
	idn = identity_matrix(FiniteField(q), n)
	one = matrix(FiniteField(q), 1, n, [1] * n)
	row_system = idn.tensor_product(one, subdivide=False)
	column_system = one.tensor_product(idn, subdivide=False)
	return row_system.augment(-one.transpose(), subdivide=False), column_system.augment(-one.transpose(), subdivide=False)


# --------------------------------------------------------------
def get_linear_system(prefix, g, g_, suffix, embed=lambda x: x):
	return embed(prefix.augment(g, subdivide=False)).tensor_product(embed((-g_.transpose()).augment(suffix, subdivide=False)), subdivide=False)


# ------------------------------------------------------------------------
def get_linear_system_transpose(prefix, g, g_, suffix, embed=lambda x: x):
	return embed((-g.transpose()).augment(suffix, subdivide=False)).tensor_product(embed(prefix.augment(g_, subdivide=False)), subdivide=False)


# -----------------------------------------------------
def get_reduced_system(n, k, q , system_, column, row=0):
	to_delete = [(i * n + j) for i in range(0, n) for j in range(0, n) if (i == row or j == column)]
	system_ij = system_.delete_columns(to_delete).augment(system_[:,n*row + column], subdivide=False)
	return system_ij


# -------------------------------------
def task(n, k, q, system, row, column):
	mtrx = get_reduced_system(n, k, q , system, column, row=row)
	# if mtrx.rank() == mtrx[:,:-1].rank(): # Rouché–Capelli Theorem
	if mtrx.rank() < 2 * k * (n - k): # This equivalent to Rouché–Capelli Theorem but 2x faster
		return column
	else:
		return None


# -------------------------------------------------------
def solve_reduced_system(n, k, q, M, M_, N, N_, columns):
	# Concerning ILCE and 2LCE
	assert(len(columns) == n)
	size = sum([len(element) for element in columns])
	assert(2 * k * (n - k) >= size)
	PolyRing = PolynomialRing(FiniteField(q), size, names="x")
	varsRing = [PolyRing.gen(i) for i in range(size)] + [1]
	
	Q = zero_matrix(PolyRing, n, n)
	element = -1
	for row in range(0, n, 1):
		for column in columns[row]:
			element += 1
			Q[row, column] = varsRing[element]

	Q11 = Q[:k,:k]
	Q12 = Q[:k,k:]
	Q21 = Q[k:,:k]
	Q22 = Q[k:,k:]
	reduced_system_eqs = Q12 + M * Q22 - Q11 * M_ - M * Q21 * M_
	reduced_system_eqs = reduced_system_eqs.stack(Q12 + N * Q22 - Q11 * N_ - N * Q21 * N_)
	reduced_system = zero_matrix(FiniteField(q), 2 * k * (n - k) + 1, size + 1)

	row = -1
	for i in range(0, 2 * k, 1):
		for j in range(0, n - k, 1):
			row += 1
			coefs = reduced_system_eqs[i,j].coefficients()
			monos = reduced_system_eqs[i,j].monomials()
			positions = [varsRing.index(monos_k) for monos_k in monos]
			for pos in range(0, len(positions), 1):
				reduced_system[row, positions[pos]] = coefs[pos]

	# add one extra equation determined by normalized monomial (non-entry of the first row is 1)
	row += 1
	equation = sum(Q[0]) - 1
	coefs = equation.coefficients()
	monos = equation.monomials()
	positions = [varsRing.index(monos_k) for monos_k in monos]
	for pos in range(0, len(positions), 1):
		reduced_system[row, positions[pos]] = coefs[pos]
	
	# solve reduced system
	solution = reduced_system[:,:-1].solve_right(-reduced_system[:,-1]).list()
	element = -1
	matrix_solution = zero_matrix(FiniteField(q), n, n)
	for row in range(0, n, 1):
		for column in columns[row]:
			element += 1
			matrix_solution[row, column] = solution[element]
	
	G1 = identity_matrix(FiniteField(q), k).augment(M, subdivide=False)
	G2 = identity_matrix(FiniteField(q), k).augment(N, subdivide=False)
	assert(is_monomial(matrix_solution, n))
	assert((G1 * matrix_solution).rref()[:k, k:n] == M_)
	assert((G2 * matrix_solution).rref()[:k, k:n] == N_)
	
	return matrix_solution


# ----------------------------------------------------------------
@counter
def algorithm(k, n, q, M, M_, N, N_, Parallel=False, bench=False):

	print('\nPublic matrix code generators')
	print(f'\nG₁ = (Iₖ | M) where M:\n{M}')
	print(f'\nG₁\' = (Iₖ | M\') where M\':\n{M_}')

	print(f'\nG₂ = (Iₖ | N) where N:\n{N}')
	print(f'\nG₂\' = (Iₖ | N\') where N\':\n{N_}')

	# starting the monitoring
	tracemalloc.start()
	prefix = identity_matrix(FiniteField(q), k)
	suffix = identity_matrix(FiniteField(q), n - k)

	# Procedure starts below
	time_start = timer()

	system = matrix(FiniteField(q), 0, n**2)
	system = system.stack(get_linear_system(prefix, M, M_, suffix))
	system = system.stack(get_linear_system(prefix, N, N_, suffix))
	# system = system.augment(zero_matrix(FiniteField(q), 2 * k * (n - k), 1))
	columns = []

	time_middle_1 = timer()
	if not Parallel:
		# Sequential approach (this is useful for debugging)
		for row in range(0, n, 1):
			# add one extra equation determined by normalized monomial (non-zero entry of the is assumed to be 1)
			# system_ = system.stack(row_system[row])
			columns.append([])
			for guess in range(0, n, 1):
				# add one extra equation determined by normalized monomial (non-zero entry of the is assumed to be 1)
				if not task(n, k, q, system, row, guess) is None:
					columns[-1].append(guess)
			print(f'{CYELLOW}[{row}]{CEND}: one column from the {CRED}{columns[-1]}{CEND}-th is different from zero')
	else:
		# Parallel approach
		n_cores = mp.cpu_count()
		print(f'\n#(total cores): {n_cores}')
		with mp.Pool(n_cores // 2) as pool:
			print(f'#(used cores):  {pool._processes}\n')
			for row in range(0, n, 1):
				# prepare arguments for reach call to target function
				# add two extra equation determined by normalized monomial (non-zero entry of the is assumed to be 1)
				inputs = [(n, k, q, system, row, guess) for guess in range(0, n, 1)]
				# call the function for each item in parallel with multiple arguments
				columns.append(list(filter(lambda input: not input is None, [result for result in pool.starmap(task, inputs)])))
				print(f'{CYELLOW}[{row}]{CEND}: one column from the {CRED}{columns[-1]}{CEND}-th is different from zero')

	time_middle_2 = timer()
	number_of_guesses = sum([len(column) for column in columns])
	number_of_guesses_per_row = number_of_guesses / n

	if number_of_guesses > 2 * k * (n - k) or number_of_guesses == 0:
		# number_of_guesses == 0 occurs for k != n/2 because of the two extra equations. In that setting we get rank equals 2k(n-k)
		first = algorithm.count == 1
		if bench:
			to_csv_file_header(f'experiments/EXC_n{n}_q{q}.csv', label=first)
		# stopping the library
		tracemalloc.stop()
		return None
	R = solve_reduced_system(n, k, q, M, M_, N, N_, columns)
	
	time_end = timer()
	
	mem_start, mem_peak = tracemalloc.get_traced_memory()
	memory = (mem_peak - mem_start) / (1024.0 * 1024.0)
	print(f'\nMemory usage:\t\t\t{memory} gigabytes')
	print(f'Average #(variables per row):\t{number_of_guesses_per_row}')
	print(f'Total number of variables:\t{number_of_guesses} = {number_of_guesses_per_row}⋅n')
	print(f'Elapsed time (get linear sys):\t{time_middle_1 - time_start} seconds')
	print(f'Elapsed time (filter process):\t{time_middle_2 - time_middle_1} seconds')
	print(f'Elapsed time (recover matrix):\t{time_end - time_middle_2} seconds')
	print(f'Elapsed time (total):\t\t{time_end - time_start} seconds\n')
	
	# stopping the library
	tracemalloc.stop()
	if bench:
		runtime = time_end - time_start
		first = algorithm.count == 1
		to_csv_file(f'experiments/EXC_n{n}_q{q}.csv', {'n':n, 'q':q, '# variables':number_of_guesses, 'memory':memory, 'runtime':runtime}, label=first)

	return R

