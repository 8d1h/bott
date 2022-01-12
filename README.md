This is a Sage library for computing integrals using Bott's residue formula. Currently the main usage is to compute Chern numbers and Fujiki constants for K3^[n], following [Ellingsrud--Göttsche--Lehn](https://arxiv.org/abs/math/9904095), and generalized Kummer varieties, following [Nieper-Wißkirchen](https://arxiv.org/abs/math/0204197). The data for OG6 and OG10 are also included.

To accelerate the computation, multithreading is used (`bott.nthreads(n)` can be used to set the number of threads); low-level integer arithmetic is done in C using `cython`.

## Examples
```
sage: from bott import *
sage: hilb_K3(5).chern_numbers()
{[10]: 176256,
 [8, 2]: 1774080,
 [6, 4]: 5075424,
 [6, 2, 2]: 12168576,
 [4, 4, 2]: 21921408,
 [4, 2, 2, 2]: 52697088,
 [2, 2, 2, 2, 2]: 126867456}
sage: kummer(5).chern_numbers()
{[10]: 2592,
 [8, 2]: 142560,
 [6, 4]: 979776,
 [6, 2, 2]: 3141504,
 [4, 4, 2]: 8141472,
 [4, 2, 2, 2]: 26220672,
 [2, 2, 2, 2, 2]: 84478464}
sage: hilb_K3(2).integral(sqrt_todd(4))
25/32
```

## Installation
Run `sage --python setup.py install --user` in the root directory where the file `setup.py` is.

Note that `cython` and `gmpy2` are needed for the package to build. They are usually already available if you have Sage installed.

To uninstall, run `sage --pip uninstall bott`.
