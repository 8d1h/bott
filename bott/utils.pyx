# distutils: libraries = gmp
from gmpy2 cimport *
import numpy as np

cdef extern from "gmp.h":
    void mpz_init(mpz_t)
    void mpz_init_set_si(mpz_t, long)
    void mpz_add(mpz_t, mpz_t, mpz_t)
    void mpz_mul(mpz_t, mpz_t, mpz_t)
    void mpz_mul_si(mpz_t, mpz_t, long)

import_gmpy2()

# depth-first search enumeration
# two versions depending on the type of w
cdef dfs_mpz(long k, long n, mpz[:] w, mpz ans, mpz[:] pp):
    cdef long m
    if k < 1:
        mpz_add(ans.z, ans.z, pp[0].z)
    else:
        for m in range(k, n+1):
            mpz_mul(pp[k-1].z, pp[k].z, w[m-1].z)
            dfs_mpz(k-1, m-1, w, ans, pp)

cdef dfs_si(long k, long n, long[:] w, mpz ans, mpz[:] pp):
    cdef long m
    if k < 1:
        mpz_add(ans.z, ans.z, pp[0].z)
    else:
        for m in range(k, n+1):
            mpz_mul_si(pp[k-1].z, pp[k].z, w[m-1])
            dfs_si(k-1, m-1, w, ans, pp)

cpdef chern(long k, list w):
    """
    Given a list w of integers, compute the sum of the products of all its
    k-combinations. Written in Cython with gmpy2 to improve performance.

    TESTS::

        sage: from bott.utils import chern
        sage: chern(8, list(range(1001, 1031)))
        6618840538644818335837329859515
        sage: chern(1, [2^100])
        1267650600228229401496703205376
    """
    cdef long i
    cdef mpz ans = GMPy_MPZ_New(NULL)
    cdef mpz[:] pp = np.array([GMPy_MPZ_New(NULL) for i in range(k+1)])
    cdef long[:] w_si = np.empty(len(w), dtype=int)
    # initialization
    mpz_init(ans.z)
    for i in range(k):
        mpz_init(pp[i].z)
    mpz_init_set_si(pp[k].z, 1)
    try:
        # check if every element of w fits as long
        for i in range(len(w)):
            w_si[i] = w[i]
        dfs_si(k, len(w), w_si, ans, pp)
        return int(ans)
    except:
        # otherwise convert them to mpz
        return chern_mpz(k, w, ans, pp)

cdef chern_mpz(long k, list w, mpz ans, mpz[:] pp):
    cdef long i
    cdef mpz[:] w_mpz = np.array([GMPy_MPZ_From_mpz(mpz(w[i]).z) for i in range(len(w))])
    dfs_mpz(k, len(w), w_mpz, ans, pp)
    return int(ans)
