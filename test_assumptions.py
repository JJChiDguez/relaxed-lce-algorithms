
import argparse

# SageMath imports
from sage.all import (
    randint,
    matrix,
    floor,
    GF,
    rank,
    RR,
    identity_matrix,
    random_matrix,
    zero_matrix,
)

def sample_permutation_matrix(n, q):

    P = zero_matrix(GF(q), n, n)

    a = [i for i in range(0, n)]
    for i in range(n-1, 0, -1):
        j = randint(0, i)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    for i in range(0, n):
        P[i,a[i]] = 1

    return P

def sample_monomial_matrix(n, q):

    P = zero_matrix(GF(q), n, n)

    a = [i for i in range(0, n)]
    for i in range(n-1, 0, -1):
        j = randint(0, i)
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

    for i in range(0, n):
        P[i,a[i]] = randint(1,q-1)

    return P

def get_system(n,k,q, G, G_):

    M_ = G_[:,k:n]
    System = G.tensor_product((-M_.transpose()).augment(identity_matrix(GF(q),n-k)))

    return System


def get_lce(n,k,q,Q):

    while(1):
        G = identity_matrix(GF(q),k).augment(random_matrix(GF(q), k, n-k))
        G_ = (G*Q).rref()
        if G_[0:k,0:k] != 1:
            continue
        break

    return G_, G

def get_ilce(n,k,q,Q):

    while(1):
        G = identity_matrix(GF(q),k).augment(random_matrix(GF(q), k, n-k))
        G_ = (G*Q).rref()
        if G_[0:k,0:k] != 1:
            continue
        Gi_ = (G*(Q)**(-1)).rref()
        if Gi_[0:k,0:k] != 1:
            continue
        break

    return G_, Gi_, G

def test_rank_lce(n, k, q, N_trials):

    t = floor(n**2/(k*(n-k))) + 1

    Q = sample_monomial_matrix(n, q)

    not_full = 0
    for _ in range(N_trials):
        G_, G = get_lce(n, k, q, Q)
        S = get_system(n,k,q, G, G_)

        for i in range(1,t):
            G_, G = get_lce(n, k, q, Q)
            S_ = get_system(n,k,q, G, G_)
            S = S.stack(S_)
        if (rank(S) != n**2 -1):
            not_full = not_full + 1
            # print(rank(S))
    print(f"LCE: n {n}, k {k}, q {q}, matrices not full rank: {not_full} / {N_trials} = {RR(not_full/N_trials)}")


def test_rank_ilce(n, k, q, N_trials):

    t = floor(n**2/(2*k*(n-k))) + 1

    Q = sample_monomial_matrix(n, q)

    not_full = 0
    for _ in range(N_trials):
        G_, Gi_, G = get_ilce(n, k, q, Q)
        S_1 = get_system(n,k,q, G, G_)
        S_2 = get_system(n,k,q, Gi_, G)
        S = S_1.stack(S_2)

        for i in range(1,t):
            G_, Gi_, G = get_ilce(n, k, q, Q)
            S_1 = get_system(n,k,q, G, G_)
            S_2 = get_system(n,k,q, Gi_, G)
            S = S.stack(S_1).stack(S_2)
        if (rank(S) != n**2 -1):
            not_full = not_full + 1
            # print(rank(S))
    print(f"ILCE: n {n}, k {k}, q {q}, matrices not full rank: {not_full} / {N_trials} = {RR(not_full/N_trials)}")


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('-mode', type=str, required=True, help='Mode to run (e.g., LCE or ILCE)')
    parser.add_argument('-n', type=int, required=True, help='Code lengtht n')
    parser.add_argument('-k', type=int, required=True, help='Code dimension k')
    parser.add_argument('-q', type=int, required=True, help='Modulo q')
    parser.add_argument('-N', type=int, required=True, help='Number of trials N')

    args = parser.parse_args()

    n = args.n
    k = args.k
    q = args.q
    N = args.N


    if args.mode == 'LCE':
        test_rank_lce(n, k, q, N)
    elif args.mode == 'ILCE':
        test_rank_ilce(n, k, q, N)
    else:
        print(f"Unknown mode")

if __name__ == "__main__":
    main()












