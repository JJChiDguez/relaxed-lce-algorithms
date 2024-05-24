# Please README


## Requirements

1. **SageMath** installed

## List of input arguments required

```bash
usage: test_XXX.py [-h] [-k CODE_DIMENSION] [-m CODE_SIZE_M] -n CODE_SIZE_N -q PRIME [-b]

Parses command.

options:
  -h, --help            show this help message and exit
  -k CODE_DIMENSION, --code_dimension CODE_DIMENSION
                        code dimension
  -m CODE_SIZE_M, --code_size_m CODE_SIZE_M
                        code size (m)
  -n CODE_SIZE_N, --code_size_n CODE_SIZE_N
                        code size (n)
  -q PRIME, --prime PRIME
                        Field characteristic
  -b, --benchmark       Benchmark
```

## The following scripts correspond to Section 3

```bash
sage -python test_lemma1.py -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME
sage -python test_lemma2.py -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME
sage -python test_corollary1.py -m CODE_SIZE_M -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME
sage -python test_corollary2.py -m CODE_SIZE_M -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME
```

## The following scripts correspond to Section 4

```bash
sage -python test_section4.1.py -n CODE_SIZE_N -q PRIME

# Below example correspond with self-dual 2-PCE instances
sage -python test_section4.1_selfdual.py

sage -python test_section4.2.py -n CODE_SIZE_N -q PRIME

# Below example correspond with self-dual IPCE instances
sage -python test_section4.2_selfdual.py
```

### Concerning the experimental validation of the assumptions

```bash
% sage -python test_assumptions.py -h
usage: test_assumptions.py [-h] -mode MODE -n N -k K -q Q -N N

Process some integers.

options:
  -h, --help  show this help message and exit
  -mode MODE  Mode to run (e.g., LCE or ILCE)
  -n N        Code lengtht n
  -k K        Code dimension k
  -q Q        Modulo q
  -N N        Number of trials N
```

## Remarks

The flag option `-b` creates a file with extension `.CSV`, which includes data concerning 25 random executions.


## License

Apache License Version 2.0, January 2004