This is a Sage library for computing integrals using Bott's residue formula. Currently the main usage is to compute Chern numbers for K3^[n], following [Ellingsrud--Göttsche--Lehn](https://arxiv.org/abs/math/9904095).

To accelerate the computation, multithreading is used (`bott.nthreads(n)` can be used to set the number of threads); low-level integer arithmetic is done in C using `cython`.

```
sage: from bott import hilb_K3
sage: hilb_K3(5).chern_numbers()
{[10]: 176256,
 [8, 2]: 1774080,
 [6, 4]: 5075424,
 [6, 2, 2]: 12168576,
 [4, 4, 2]: 21921408,
 [4, 2, 2, 2]: 52697088,
 [2, 2, 2, 2, 2]: 126867456}
```

## Installation
Run `sage --python setup.py install --user` in the root directory where the file `setup.py` is.

Note that `cython` and `gmpy2` are needed for the package to build. They are usually already available if you have Sage installed.

To uninstall, run `sage --pip uninstall bott`.
