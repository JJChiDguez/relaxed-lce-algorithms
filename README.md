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
sage -python test_lemma2.py -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME
sage -python test_corollary2b.py -m CODE_SIZE_M -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME

sage -python test_lemma3.py -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME
sage -python test_corollary4b.py -m CODE_SIZE_M -n CODE_SIZE_N -k CODE_DIMENSION -q PRIME
```


## The following scripts correspond to Section 4

```bash
sage -python test_theorem1.py -n CODE_SIZE_N -q PRIME
sage -python test_theorem1_selfdual.py -n CODE_SIZE_N -q PRIME

sage -python test_theorem2.py -n CODE_SIZE_N -q PRIME
sage -python test_theorem2_IPCE.py -n CODE_SIZE_N -q PRIME
sage -python test_theorem2_selfdual.py -n CODE_SIZE_N -q PRIME
sage -python test_theorem2_seldual_IPCE.py -n CODE_SIZE_N -q PRIME
```


## Remarks

The flag option `-b` creates a file with extension .CSV, which includes data concerning ten random executions.


## License

Apache License Version 2.0, January 2004