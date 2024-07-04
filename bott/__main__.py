from sage.all import *
from types import MethodType

from .utils import chern

class Nthreads:
    def __init__(self):
        self.n = 8
Nthreads = Nthreads()
def nthreads(n):
    """
    Set the number of threads to use when computing using Bott's formula.
    """
    Nthreads.n = int(n)

class Cached:
    def __init__(self):
        self.K3n = load("K3n")
        self.Kumn = load("Kumn")
Cached = Cached()

def list_to_partition(l):
    if isinstance(l, Partition):
        return l
    if isinstance(l, list):
        l = l.copy()
        l.sort(reverse=True)
        return Partition(l)
    raise ValueError("Cannot convert "+str(l)+" to a partition")

class TnVariety:
    """
    A variety with a torus action for computing with Bott's formula.
    """
    def __init__(self, n, points):
        self.dim = n
        self.points = points

    def tangent_bundle(self):
        return self.T

    def cotangent_bundle(self):
        return self.T.dual()

    def dimension(self):
        return self.dim

    def fixed_points(self):
        """
        Return the list of fixed points of the torus action.
        """
        return list(self.points)

    def chern_numbers(self):
        """
        Compute a dictionary of Chern numbers using Bott's formula.

        The computation is carried out with multithreading to improve the
        performance. The number of threads can be configured using
        `nthreads(n)`.

        EXAMPLES::

            sage: from bott import Pn
            sage: print(Pn(2).chern_numbers())
            {[2]: 3, [1, 1]: 9}
        """
        return self.T.integrals(Partitions(self.dim))

    def cobordism_class(self):
        """
        Return the cobordism class.

        EXAMPLES::

            sage: from bott import Pn
            sage: Pn(2).cobordism_class()
            Cobordism Class of dim 2
        """
        return CobordismClass(self.dim, self.chern_numbers(), check=False)

    def O(self, n=0):
        """
        Return the line bundle O(n) as a TnBundle (if it is defined).

        EXAMPLES::

            sage: from bott import Pn
            sage: Pn(3).O(1)
            TnBundle of rank 1 on TnVariety of dim 3
        """
        if n == 0:
            return TnBundle(self, 1, lambda p: [0])
        if not hasattr(self, "O1"):
            raise ValueError("No polarization defined")
        return TnBundle(self, 1, lambda p: [n * self.O1.weight(p)[0]])

    def integrals(self, Fs, c):
        # a list of bundles Fs and a polynomial in Chern classes
        partitions = {}
        for monom in c.monomials():
            if monom.degree() == self.dim:
                d = monom.exponents()[0]
                pp = []
                k = 0
                for F in Fs:
                    pp.append(Partition(exp=d[k:k+rank(F)]))
                    k += rank(F)
                partitions[tuple(pp)] = c.monomial_coefficient(monom)
        ans = self.__integrals(Fs, partitions.keys())
        return sum(a * ans[p] for (p, a) in partitions.items())

    def __integrals(self, Fs, partitions):
        idx = [set() for _ in Fs]
        for p in partitions:
            for i in range(len(Fs)):
                idx[i] = idx[i].union(set(p[i]))

        @parallel(Nthreads.n)
        def compute(points):
            ans = {p:0 for p in partitions}
            for pt in points:
                cs = [[] for _ in range(len(Fs))]
                for i in range(len(Fs)):
                    w = Fs[i].weight(pt)
                    cs[i] = [1] + [chern(k, w) if k in idx[i] else 0 for k in range(1, len(w)+1)] + [0 for k in range(len(w), self.dim)]
                ctop = chern(self.dim, self.T.weight(pt))
                for p in ans.keys():
                    ans[p] += prod(cs[i][k] for i in range(len(p)) for k in p[i]) * QQ((1, ctop))
            return ans

        each = max(len(self.points) // Nthreads.n, 10)
        num = len(self.points) // each
        c = compute([self.points[i*each:(i+1)*each] for i in range(num)] + [self.points[num*each:]])
        ans = {p: 0 for p in partitions}
        for (_, ci) in c:
            for p in ans.keys():
                ans[p] += ci[p]
        return ans

    def __mul__(self, other):
        """
        Return the product of two `TnVariety`.

        TESTS::

            sage: from bott import Pn
            sage: Pn(2) * Pn(3)
            TnVariety of dim 5 with 12 fixed points
        """
        if not isinstance(other, TnVariety):
            raise TypeError(str(other) + " is not a TnVariety")
        n = self.dim + other.dim
        points = cartesian_product([self.points, other.points])
        weight = lambda p: self.T.weight(p[0]) + other.T.weight(p[1])
        X = TnVariety(n, points)
        X.T = TnBundle(X, n, weight)
        try:
            X.w = [self.w, other.w]
        except:
            pass
        return X

    def __repr__(self):
        return "TnVariety of dim " + str(self.dim) + " with " + str(len(self.points)) + " fixed points"

class TnBundle:
    """
    An equivariant bundle for a torus action.
    """
    def __init__(self, X, r, weight):
        self.parent = X
        self.r = r
        self.weight = weight

    def rank(self):
        return self.r

    def integral(self, c=None):
        X = self.parent
        if c == None: # top Chern class
            p = Partition([X.dim])
            return self.integrals([p])[p]
        if isinstance(c, Partition): # Chern class as Partition
            return self.integrals([c])[c]
        if isinstance(c, list): # Chern class as list
            c = list_to_partition(c)
            return self.integrals([c])[c]
        # Chern class as polynomial
        return self.parent.integrals([self], c)

    def integrals(self, partitions):
        ans = self.parent._TnVariety__integrals([self], [(p,) for p in partitions])
        return {p: ans[(p,)] for p in partitions}

    def dual(self):
        return TnBundle(self.parent, rank(self), lambda p: [-wi for wi in self.weight(p)])

    def det(self):
        return TnBundle(self.parent, 1, lambda p: [sum(self.weight(p))])

    def symmetric_power(self, k):
        return TnBundle(self.parent, binomial(rank(self)+k-1,k), lambda p: [sum(c) for c in sym(self.weight(p), k)])

    def exterior_power(self, k):
        return TnBundle(self.parent, binomial(rank(self), k), lambda p: [sum(c) for c in comb(self.weight(p), k)])

    def chi(self):
        """
        Return the holomorphic Euler characteristic, using
        Hirzebruch-Riemann-Roch.

        EXAMPLES::

            sage: from bott import Pn
            sage: Pn(3).cotangent_bundle().chi()
            -1
        """
        X = self.parent
        R = PolynomialRing(QQ, rank(self)+dim(X), ["c"+str(i+1) for i in range(rank(self))] + ["d"+str(i+1) for i in range(X.dim)], order=TermOrder("wdeglex", list(range(1, rank(self)+1)) + list(range(1, X.dim+1))))
        ch = rank(self) + capped_logg(R(sum(R.gens()[:rank(self)])), dim(X))
        td = todd(dim(X))
        td = hom(parent(td), R, R.gens()[rank(self):])(td)
        return X.integrals([self, X.T], ch * td) # HRR

    def __check_other(self, other):
        if not isinstance(other, TnBundle):
            raise TypeError(str(other) + " is not a TnBundle")
        if not other.parent == self.parent:
            raise ValueError("Parents do not agree")

    def __add__(self, other):
        self.__check_other(other)
        return TnBundle(self.parent, rank(self) + rank(other), lambda p: self.weight(p) + other.weight(p))

    def __mul__(self, other):
        self.__check_other(other)
        return TnBundle(self.parent, rank(self) * rank(other), lambda p: [u + v for u in self.weight(p) for v in other.weight(p)])

    def boxtimes(self, other):
        """
        Return the box product of self and other, that is, the tensor product
        of the respective pullbacks to the product variety.

        EXAMPLES::

            sage: from bott import Pn
            sage: Pn(1).O(1).boxtimes(Pn(1).O(2))
            TnBundle of rank 1 on TnVariety of dim 2
        """
        if not isinstance(other, TnBundle):
            raise TypeError(str(other) + " is not a TnBundle")
        return TnBundle(self.parent * other.parent, rank(self) * rank(other), lambda p: [u + v for u in self.weight(p[0]) for v in other.weight(p[1])])

    def __repr__(self):
        return "TnBundle of rank " + str(rank(self)) + " on TnVariety of dim " + str(self.parent.dim)

def comb(w, k):
    def dfs(k, n):
        if k < 1:
            yield pp[:]
        else:
            for m in range(k, n+1):
                pp[k-1] = w[m-1]
                yield from dfs(k-1, m-1)
    pp = [0] * k
    yield from dfs(k, len(w))

def sym(w, k):
    def dfs(k, n):
        if k < 1:
            yield pp[:]
        else:
            for m in range(1, n+1):
                pp[k-1] = w[m-1]
                yield from dfs(k-1, m)
    pp = [0] * k
    yield from dfs(k, len(w))

class CobordismClass:
    """
    A cobordism class represented by the dimension and the Chern numbers.

    EXAMPLES::

        sage: from bott import CobordismClass
        sage: CobordismClass(2, {Partition([2]): 24})
        Cobordism Class of dim 2

    TESTS::

        sage: from bott import CobordismClass
        sage: CobordismClass(2, {Partition([1]): 1})
        Traceback (most recent call last):
        ...
        ValueError: [1] is not a partition of 2
    """
    def __init__(self, n, chern_numbers, check=True):
        self.dim = n
        self.__cn = chern_numbers
        if check:
            for p in chern_numbers.keys():
                if p.size() != n:
                    raise ValueError(str(p) + " is not a partition of " + str(n))
        for p in Partitions(n):
            if not p in self.__cn.keys():
                self.__cn[p] = 0

    def dimension(self):
        return self.dim

    def chern_numbers(self, nonzero=True):
        """
        Return a dictionary of Chern numbers. Only non-zero ones are shown by
        default.

        EXAMPLES::

            sage: from bott import hilb_K3
            sage: hilb_K3(1).chern_numbers()
            {[2]: 24}
        """
        if nonzero:
            return {p: v for (p, v) in self.__cn.items() if v != 0}
        return self.__cn

    def chern_number(self, p):
        """
        Return a single Chern number.

        EXAMPLES::

            sage: from bott import hilb_K3
            sage: hilb_K3(2).chern_number([4])
            324
        """
        p = list_to_partition(p)
        if p.size() != self.dim:
            raise ValueError(str(p) + " is not a partition of " + str(self.dim))
        return self.__cn[p]

    def integral(self, c):
        """
        Compute the integral of a polynomial in Chern classes.
        """
        ans = 0
        for monom in c.monomials():
            if monom.degree() == self.dim:
                d = monom.exponents()[0]
                p = Partition(exp=d)
                ans += c.monomial_coefficient(monom) * self.__cn[p]
        return ans

    def chern_characters(self, nonzero=True):
        """
        Return a dictionary of Chern numbers in Chern characters. Only
        non-zero ones are shown by default.

        EXAMPLES::

            sage: from bott import hilb_K3
            sage: hilb_K3(1).chern_characters()
            {[2]: -24}
        """
        ans = {}
        ch = by_degree(chern_character(self.dim), self.dim)
        for p in Partitions(self.dim):
            ans[p] = self.integral(prod(ch[k] for k in p))
        if nonzero:
            return {p: v for (p, v) in ans.items() if v != 0}
        return ans

    def chern_character(self, p):
        """
        Return a single Chern number in Chern characters.

        EXAMPLES::

            sage: from bott import hilb_K3
            sage: hilb_K3(2).chern_character([4])
            15
        """
        p = list_to_partition(p)
        if p.size() != self.dim:
            raise ValueError(str(p) + " is not a partition of " + str(self.dim))
        ch = by_degree(chern_character(self.dim), self.dim)
        return self.integral(prod(ch[k] for k in p))

    def __mul__(self, other):
        """
        Return the product of two `CobordismClass`.

        TESTS::

            sage: from bott import hilb_K3
            sage: hilb_K3(1) * hilb_K3(1)
            Cobordism Class of dim 4
            sage: hilb_K3(0) * hilb_K3(0)
            Cobordism Class of dim 0
        """
        if not isinstance(other, CobordismClass):
            raise TypeError(str(other) + " is not a CobordismClass")
        m, n = self.dim, other.dim
        if m > n:
            return other * self
        d = _product(m, n)
        pp = Partitions(m+n)
        ans = {p: 0 for p in pp}
        for p in pp:
            for (p1p2, v) in d[p].items():
                p1, p2 = p1p2
                ans[p] += v * self.chern_number(p1) * other.chern_number(p2)
        return CobordismClass(m + n, ans)

    def __eq__(self, other):
        if isinstance(other, CobordismClass):
            return dim(self) == dim(other) and self.chern_numbers() == other.chern_numbers()
        return False

    def __repr__(self):
        return "Cobordism Class of dim " + str(self.dim)

@cached_function
def _product(m, n):
    R = PolynomialRing(QQ, m+n, ["c"+str(i) for i in range(m)] + ["d"+str(i) for i in range(n)], order=TermOrder("wdeglex", list(range(1,m+1)) + list(range(1,n+1))))
    TX = R(1)+sum(R.gens()[0:m])
    TY = R(1)+sum(R.gens()[m:m+n])
    cTXY = TX * TY
    c = by_degree(cTXY, m+n)
    pp = Partitions(m+n)
    ans = {p: {} for p in pp}
    for p in pp:
        cp = prod((c[i] for i in p), z=R(1))
        for monom in cp.monomials():
            d = monom.exponents()[0]
            if sum(d[i] * (i+1) for i in range(m)) == m:
                p1 = Partition(exp=d[:m])
                p2 = Partition(exp=d[m:])
                ans[p][(p1,p2)] = cp.monomial_coefficient(monom)
    return ans

def Pn(n, w=None):
    """
    Projective space of dimension n, with a torus action.

    EXAMPLES::

        sage: from bott import Pn
        sage: Pn(3)
        TnVariety of dim 3 with 4 fixed points
    """
    return grassmannian(1, n+1, w=w)

def grassmannian(k, n, w=None):
    """
    Grassmannian Gr(k, n), with a torus action.

    EXAMPLES::

        sage: from bott import grassmannian
        sage: grassmannian(2, 4)
        TnVariety of dim 4 with 6 fixed points
        sage: grassmannian(2, 4).bundles
        [TnBundle of rank 2 on TnVariety of dim 4,
         TnBundle of rank 2 on TnVariety of dim 4]
        sage: grassmannian(2, 4).bundles[0].dual().symmetric_power(3).integral()
        27
    """
    G = flag(k, n, w=w)
    G.O1 = G.bundles[0].det().dual()
    return G

def flag(*dims, w=None):
    """
    Flag variety with a torus action.

    EXAMPLES::

        sage: from bott import flag
        sage: flag(1,6,10)
        TnVariety of dim 29 with 1260 fixed points
        sage: A,B,C = flag(1,6,10).bundles
        sage: integral((A*B.exterior_power(2) + A*B*C).dual())
        990
    """
    n, l = dims[-1], len(dims)
    if w == None:
        w = list(range(n))
    if w != None and not isinstance(w, list):
        raise ValueError("weight should be a list")
    ranks = [dims[0]] + [dims[i+1] - dims[i] for i in range(l-1)]
    d = sum(ranks[i] * (n - dims[i]) for i in range(l))
    def enum(i, rest):
        if i == l:
            return [[rest]]
        return [[tuple(x)] + y for x in Combinations(rest, ranks[i-1]) for y in enum(i+1, tuple([r for r in rest if not r in x]))]
    Fl = TnVariety(d, [tuple(c) for c in enum(1, range(n))])
    def closure(i):
        return lambda p: [w[j] for j in p[i]]
    Fl.bundles = [TnBundle(Fl, ranks[i], closure(i)) for i in range(l) if ranks[i] > 0]
    l = len(Fl.bundles)
    z = TnBundle(Fl, 0, lambda p: [])
    Fl.T = sum([Fl.bundles[i].dual() * sum([Fl.bundles[j] for j in range(i+1,l)], z) for i in range(l-1)], z)
    Fl.w = w
    return Fl

def _hilb(n, parts, wt):
    points = [x for pp in parts for x in cartesian_product([Partitions(x) for x in pp])]
    def weight(pt):
        w = []
        for p,(l,m) in zip(pt, wt):
            if len(p) > 0:
                b = p.conjugate()
                for s in range(len(p)):
                    j = p[s]
                    for i in range(j):
                        w.append(l*(i-j)   + m*(b[i]-s-1))
                        w.append(l*(j-i-1) + m*(s-b[i]))
        return w
    X = TnVariety(2*n, points)
    X.T = TnBundle(X, 2*n, weight)
    return X

def hilb_P2(n, w=None):
    """
    Hilbert scheme of n points on the projective plane, with a torus action.

    EXAMPLES::

        sage: from bott import hilb_P2
        sage: hilb_P2(2)
        TnVariety of dim 4 with 9 fixed points
    """
    if w == None:
        w = [0,1,n+1]
    return _hilb(n, [[a, b-a-1, n+1-b] for a,b in Combinations(n+2, 2)], [[w[k]-w[k+1], w[k]-w[k+2]] for k in GF(3)])

def hilb_P1xP1(n, w=None):
    """
    Hilbert scheme of n points on the product of two projective lines, with a
    torus action.

    EXAMPLES::

        sage: from bott import hilb_P1xP1
        sage: hilb_P1xP1(2)
        TnVariety of dim 4 with 14 fixed points
    """
    if w == None:
        w = [[0,1], [2,n+2]]
    return _hilb(n, [[a, b-a-1, c-b-1, n+2-c] for a,b,c in Combinations(n+3, 3)], [[w[0][k]-w[0][k+1], w[1][j]-w[1][j+1]] for k in GF(2) for j in GF(2)])

def by_degree(x, n):
    c = [x.parent(0)] * (n+1)
    for monom in x.monomials():
        d = monom.degree()
        if d <= n:
            c[d] += x.monomial_coefficient(monom) * monom
    return c

def capped_log(x, n):
    if n == 0:
        return x.parent(0)
    e = by_degree(x, n)
    p = [e[1]] + [0] * (n-1)
    for i in range(n-1):
        p[i+1] = (i+2) * e[i+2] - sum(e[j+1] * p[i-j] for j in range(i+1))
    return sum(QQ((1, i+1))*p[i] for i in range(n))

def capped_logg(x, n):
    if n == 0:
        return x.parent(0)
    e = by_degree(x, n)
    p = [-e[1]] + [0] * (n-1)
    for i in range(n-1):
        p[i+1] = -(i+2) * e[i+2] - sum(e[j+1] * p[i-j] for j in range(i+1))
    return sum(QQ(((-1)**(i+1), factorial(i+1)))*p[i] for i in range(n))

def capped_exp(x, n):
    comps = by_degree(x, n)
    p = [i * comps[i] for i in range(n+1)]
    e = [x.parent(1)] + [0] * (n)
    for i in range(n):
        e[i+1] = QQ((1, i+1)) * sum(p[j+1] * e[i-j] for j in range(i+1))
    return sum(e)

def capped_expp(x, n):
    comps = by_degree(x, n)
    p = [(-1)**(i+1) *factorial(i) * comps[i] for i in range(n+1)]
    e = [x.parent(1)] + [0] * (n)
    for i in range(n):
        e[i+1] = QQ((1, i+1)) * sum(p[j+1] * e[i-j] for j in range(i+1))
    return sum(e)

# streamline multithreaded computation of Chern numbers
# `x` is a polynomial
# `classes` specifies the cobordism classes each variable represents
def compute_sum(x, dim, classes, check = lambda _: True):
    N = parent(x).ngens()
    @parallel(Nthreads.n)
    def compute(terms):
        ans = {p: 0 for p in partitions}
        for (monom, coeff) in terms:
            if check(monom.degree()):
                d = monom.exponents()[0]
                c = prod([classes[k] for k in range(N) for _ in range(d[k])], z=Pn(0).cobordism_class())
                for p in partitions:
                    ans[p] += coeff * c.chern_number(p)
        return ans
    terms = [(monom, x.monomial_coefficient(monom)) for monom in x.monomials()]
    partitions = Partitions(dim)
    each = max(len(terms) // Nthreads.n, 10)
    num = len(terms) // each
    c = compute([terms[i*each:(i+1)*each] for i in range(num)] + [terms[num*each:]])
    ans = {p: 0 for p in partitions}
    for (_, ci) in c:
        for p in partitions:
            ans[p] += ci[p]
    return ans

class HKCobordismClass(CobordismClass):
    deformation_type = ''
    def __eq__(self, other):
        if isinstance(other, HKCobordismClass):
            return dim(self) == dim(other) and self.chern_numbers() == other.chern_numbers() and self.fujiki_constants() == other.fujiki_constants()
        return False
    @cached_method
    def fujiki_constants(self):
        """
        Return the Fujiki constants of Chern classes on a hyperkähler manifold.

        EXAMPLES::

            sage: from bott import hilb_K3, kummer
            sage: print(hilb_K3(2).fujiki_constants())
            {[]: 3, [2]: 30, [4]: 324, [2, 2]: 828}
            sage: print(kummer(2).fujiki_constants())
            {[]: 9, [2]: 54, [4]: 108, [2, 2]: 756}
        """
        n, typ = dim(self)//2, self.deformation_type
        if typ == "hilb_K3":
            g = universal_genus(2*n)
            Omega = parent(g).base_ring()
            P = [Omega(0)] + list(Omega.gens())
            Rt = PolynomialRing(Omega, "t")
            t = Rt.gen()
            A, B = Rt(1), Rt(1)
            for k in range(1, n+1):
                R = PolynomialRing(Omega, 2*k+1, ["c"+str(i+1) for i in range(2*k)] + ["E"], order=TermOrder("wdeglex", list(range(1, 2*k+1)) + [1]))
                gk = universal_genus(2*k)
                c = hom(parent(gk), R, R.gens()[:-1])(gk) * capped_exp(R.gens()[-1], 2*k)
                # pick a random weight w
                Ok = to_P2k(Pn(2, w=[0,12,1321]).O(), k)
                P2k = Ok.parent
                A += t**k * P2k.integrals([P2k.T, Ok], c)
                # pick random weights w
                Ok = to_P1xP1k((Pn(1, w=[0,121]) * Pn(1, w=[0,213])).O(), k)
                P1xP1k = Ok.parent
                B += t**k * P1xP1k.integrals([P1xP1k.T, Ok], c)
            C = capped_exp(-16 * capped_log(A, n) + 18 * capped_log(B, n), n)
            Cn = by_degree(C.monomial_coefficient(t**n), 2*n)
            c = [0] + list(parent(g).gens())
            ans = {}
            coeffs_in_Pn = lambda x, m: [x.monomial_coefficient(Omega(prod(P[k] for k in p))) for p in Partitions(m)]
            for m in range(0, n+1):
                M = matrix([coeffs_in_Pn(g.monomial_coefficient(g.parent(prod(c[2*k] for k in p))), 2*m) for p in Partitions(m)])
                coeffs = M.solve_left(vector(coeffs_in_Pn(Cn[2*m], 2*m)))
                for (q, v) in zip(Partitions(m), coeffs):
                    q = Partition([2*qi for qi in q])
                    if m == 0 and n == 1:
                        ans[q] = QQ(1)
                    else:
                        ans[q] = v * factorial(2*(n-m)) / (2-2*n)**(n-m)
            return ans
        elif typ == "kummer":
            g = universal_genus(2*(n+1))
            Omega = parent(g).base_ring()
            P = [Omega(0)] + list(Omega.gens())
            Rt = PolynomialRing(Omega, "t")
            t = Rt.gen()
            Ap, Am, A = Rt(1), Rt(1), Rt(1)
            for k in range(1, n+2):
                R = PolynomialRing(Omega, 2*k+1, ["c"+str(i+1) for i in range(2*k)] + ["E"], order=TermOrder("wdeglex", list(range(1, 2*k+1)) + [1]))
                # pick a random weight w
                Ok = to_P2k(Pn(2, w=[0,12,1321]).O(), k)
                P2k = Ok.parent
                gk = universal_genus(2*k, twist=1)
                c = hom(parent(gk), R, R.gens()[:-1])(gk) * capped_exp(R.gens()[-1], 2*k)
                Ap += t**k * P2k.integrals([P2k.T, Ok], c)
                gk = universal_genus(2*k, twist=-1)
                c = hom(parent(gk), R, R.gens()[:-1])(gk) * capped_exp(R.gens()[-1], 2*k)
                Am += t**k * P2k.integrals([P2k.T, Ok], c)
                gk = universal_genus(2*k)
                c = hom(parent(gk), R, R.gens()[:-1])(gk) * capped_exp(R.gens()[-1], 2*k)
                A += t**k * P2k.integrals([P2k.T, Ok], c)
            K = QQ(((n+1)**2, 9)) * by_degree(capped_log(Ap, n+1) + capped_log(Am, n+1) - 2*capped_log(A, n+1), n+1)[n+1].coefficients()[0]
            Kn = by_degree(K, 2*n)
            c = [0] + list(parent(g).gens())
            ans = {}
            coeffs_in_Pn = lambda x, m: [x.monomial_coefficient(Omega(prod(P[k] for k in p))) for p in Partitions(m)]
            for m in range(0, n+1):
                M = matrix([coeffs_in_Pn(g.monomial_coefficient(g.parent(prod(c[2*k] for k in p))), 2*m) for p in Partitions(m)])
                coeffs = M.solve_left(vector(coeffs_in_Pn(Kn[2*m], 2*m)))
                for (q, v) in zip(Partitions(m), coeffs):
                    q = Partition([2*qi for qi in q])
                    if m == 0 and n == 1:
                        ans[q] = QQ(1)
                    else:
                        ans[q] = v * factorial(2*(n-m)) / (-2-2*n)**(n-m)
            return ans
        elif typ == "OG6":
            ans = {Partition(p): QQ(v) for (p, v) in [
                ([], 60),
                ([2], 288),
                ([4], 480),
                ([2,2], 1920)
            ]}
            for k in self.chern_numbers().keys():
                ans[k] = self.chern_number(k)
            return ans
        elif typ == "OG10":
            ans = {Partition(p): QQ(v) for (p, v) in [
                ([], 945),
                ([2], 5040),
                ([4], 13500),
                ([2,2], 32400),
                ([6], 26460),
                ([4,2], 113400),
                ([2,2,2], 272160),
                ([8], 49770),
                ([6,2], 343980),
                ([4,4], 614250),
                ([4,2,2], 1474200),
                ([2,2,2,2], 3538080)
            ]}
            for k in self.chern_numbers().keys():
                ans[k] = self.chern_number(k)
            return ans
        else:
            raise NotImplementedError

    def fujiki_constant(self, p):
        """
        Return a single Fujiki constant. Several types of inputs are accepted:

            - A partition / list: returns the Fujiki constant for the corresponding
              product of Chern classes
            - A polynomial in Chern classes
            - A polynomial in Chern classes and a variable named `q`: q will
              represent the dual of the Beauville-Bogomolov-Fujiki form

        EXAMPLES::

            sage: from bott import hilb_K3, c, td
            sage: hilb_K3(2).fujiki_constant([2])
            30
            sage: hilb_K3(2).fujiki_constant(c[2])
            30
            sage: R.<q> = parent(c[2])[]
            sage: hilb_K3(2).fujiki_constant(c[2] - 6/5*q)
            0

        TESTS::

            sage: from bott import hilb_K3, kummer
            sage: hilb_K3(1).fujiki_constant([])
            1
            sage: kummer(1).fujiki_constant([])
            1
        """
        if isinstance(p, Partition):
            return self.fujiki_constants()[p]
        if isinstance(p, list):
            return self.fujiki_constants()[list_to_partition(p)]
        # `is_homogeneous` is broken in Sage if variables are weighted
        # if not p.is_homogeneous():
        #     raise ValueError(str(p) + " is not homogeneous")
        F = self.fujiki_constants()
        ans = 0
        R = p.parent()
        if R.ngens() == 1 and R.variable_name() == 'q':
            n, typ = dim(self)//2, self.deformation_type
            assert p.degree() <= n
            if typ == 'hilb_K3':
                b = ZZ(22) if n == 1 else ZZ(23)
            elif typ == 'kummer':
                b = ZZ(22) if n == 1 else ZZ(7)
            elif typ == 'OG6':
                b = ZZ(8)
            elif typ == 'OG10':
                b = ZZ(24)
            for (l, c) in enumerate(list(p)):
                k = c.degree()
                assert k//2 + l <= n
                ans += self.fujiki_constant(c)*prod((b+2*n-k-2*i)/(1+2*n-k-2*i) for i in range(1, l+1))
            return ans
        else:
            for (c, m) in p:
                q = Partition(exp=m.degrees())
                ans += c * F[q] if q in F.keys() else 0
            return ans

@cached_function
def hilb_K3(n, cached=True):
    """
    Compute the Chern numbers of the Hilbert scheme of n points on a K3
    surface. Return a `CobordismClass` object.

    EXAMPLES::

        sage: from bott import hilb_K3
        sage: X = hilb_K3(3)
        sage: print(X.chern_numbers())
        {[6]: 3200, [4, 2]: 14720, [2, 2, 2]: 36800}

    TESTS::

        sage: from bott import hilb_K3
        sage: hilb_K3(0)
        Cobordism Class of dim 0
        sage: hilb_K3(2) == hilb_K3(2, cached=False)
        True
    """
    if 0 < n <= 13 and cached:
        X = HKCobordismClass(2*n, Cached.K3n[n][0])
        X.fujiki_constants = MethodType(lambda self: Cached.K3n[n][1], X)
    else:
        X = HKCobordismClass(2*n, hilb_surface(n, 0, 24).chern_numbers(), False)
    X.deformation_type = "hilb_K3"
    return X

def hilb_surface(n, c1c1, c2, base_ring=QQ):
    """
    Compute the Chern numbers of the Hilbert scheme of n points on a surface
    with given Chern numbers. Return a `CobordismClass` object.

    EXAMPLES::

        sage: from bott import hilb_surface
        sage: X = hilb_surface(2, 0, 24)
        sage: print(X.chern_numbers())
        {[4]: 324, [2, 2]: 828}

    The Chern numbers are given by universal polynomials in two variables. One
    can work in the symbolic ring `SR` to obtain these polynomials.

    EXAMPLES::

        sage: var("a b")
        (a, b)
        sage: X = hilb_surface(2, a, b, base_ring=SR)
        sage: print(X.chern_numbers())
        {[4]: 1/2*b^2 + 3/2*b, [3, 1]: a*b + 3*a, [2, 2]: 1/2*a^2 + b^2 + a + 21/2*b, [2, 1, 1]: a^2 + a*b + 6*a, [1, 1, 1, 1]: 3*a^2}
    """
    a, b = matrix([[9,8],[3,4]]).solve_right(vector([c1c1, c2]))
    S = PolynomialRing(base_ring, 2*n, ["a"+str(i+1) for i in range(n)]+["b"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple(range(1,n+1))+tuple(range(1,n+1))))
    A, B = gens(S)[:n], gens(S)[n:]
    HS = capped_exp(a*capped_log(S(1)+sum(A), n)+b*capped_log(S(1)+sum(B), n), n)
    P2n    = [hilb_P2(k+1).cobordism_class()    for k in range(n)]
    P1xP1n = [hilb_P1xP1(k+1).cobordism_class() for k in range(n)]
    ans = compute_sum(HS, 2*n, P2n+P1xP1n, check=lambda d: d == n)
    # in case one works in the Symbolic Ring
    if base_ring == SR:
        for p in ans.keys():
            ans[p] = ans[p].expand().simplify()
    return CobordismClass(2*n, ans)

def genus(x, taylor, n, twist=0):
    S = PowerSeriesRing(taylor[0].parent(), "t", default_prec=n+1)
    t = S.gen()
    lg = log(sum(taylor[i] * t**i for i in range(n+1))).list()
    lg = lg + [0] * (n+1-len(lg))
    comps = by_degree(x, n)
    T = twist * x.parent().gens()[0] if twist != 0 else 0 # assuming that the first gen is c1
    g = capped_exp(sum(factorial(i)*lg[i]*comps[i] for i in range(n+1)) + T, n)
    return g

def _taylor(phi, n):
    if n == 0:
        return [phi[0]]
    ans = _taylor(phi, n-1) + [0]
    S = PolynomialRing(phi[0].parent(), "h")
    chTP = S([n]+[QQ((n+1, factorial(i))) for i in range(1,n+1)])
    x = by_degree(genus(chTP, ans, n), n)[n]
    x = 0 if x == 0 else x.coefficients()[0]
    ans[n] = QQ((1, n+1)) * (phi[n] - x)
    return ans

def universal_genus(n, images=None, series=None, twist=0):
    """
    The universal genus is a polynomial in Chern classes with coefficients in
    the cobordism ring. Elements of the cobordism ring are represented as
    polynomials where each variable stands for the class of a projective space.

    EXAMPLES::

        sage: from bott import universal_genus
        sage: universal_genus(2)
        (-1/4*P1^2 + 1/3*P2)*c1^2 + (3/4*P1^2 - 2/3*P2)*c2 + 1/2*P1*c1 + 1

    The universality means every genus can be deduced from this: for a given genus
    whose images of the first n projective spaces are known, one can compute
    its total class as a polynomial in Chern classes. For example, the Todd
    genus can be obtained using a list of ones.

    EXAMPLES::

        sage: universal_genus(3, images=[QQ(1)]*3)
        1/24*c1*c2 + 1/12*c1^2 + 1/12*c2 + 1/2*c1 + 1

    Alternatively, the universal genus can be described in terms of the
    universal formal power series 1 + a1*t + a2*t^2 + ..., and any given genus
    can be obtained by specifying the values for the coefficients.

    EXAMPLES::

        sage: universal_genus(2, series=True)
        a2*c1^2 + (a1^2 - 2*a2)*c2 + a1*c1 + 1
        sage: universal_genus(2, series=[1,1/2,1/12])
        1/12*c1^2 + 1/12*c2 + 1/2*c1 + 1
    """
    if series:
        if series == True:
            S = PolynomialRing(QQ, n, ["a"+str(i+1) for i in range(n)])
            series = [S(1)] + list(S.gens())
        else:
            S = QQ
            series = [QQ(a_k) for a_k in series]
    elif images == None:
        S = PolynomialRing(QQ, n, ["P"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple(range(1,n+1))))
        series = _taylor([S(1)] + list(S.gens()), n)
    else:
        S = images[0].parent() if images else QQ
        series = _taylor([S(1)] + images, n)
    g = genus(capped_logg(total_chern(n, S), n), series, n, twist=twist)
    return g

def todd(n, base_ring=QQ):
    """
    Compute the Todd genus as a polynomial in Chern classes.

    EXAMPLES::

        sage: from bott import todd
        sage: todd(3)
        1/24*c1*c2 + 1/12*c1^2 + 1/12*c2 + 1/2*c1 + 1
    """
    return universal_genus(n, images=[base_ring(1)]*n)

def sqrt_todd(n, alpha=QQ((1,2)), base_ring=QQ):
    """
    Compute the square root of the Todd genus as a polynomial in Chern
    classes. Use `alpha` to compute other powers.

    EXAMPLES::

        sage: from bott import hilb_K3, sqrt_todd
        sage: sqrt_todd(2)
        1/96*c1^2 + 1/24*c2 + 1/4*c1 + 1
        sage: hilb_K3(2).integral(sqrt_todd(4))
        25/32
    """
    return capped_exp(alpha*capped_log(todd(n, base_ring), n), n)

def total_chern(n, base_ring=QQ):
    R = PolynomialRing(base_ring, n, ["c"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple(range(1,n+1))))
    return R(1) + sum(R.gens())

def chern_character(rank, dim=None, base_ring=QQ):
    """
    Compute the Chern character as a polynomial in Chern classes.

    EXAMPLES::

        sage: from bott import chern_character
        sage: chern_character(3)
        1/6*c1^3 - 1/2*c1*c2 + 1/2*c3 + 1/2*c1^2 - c2 + c1 + 3
        sage: chern_character(1, 4)
        1/24*c1^4 + 1/6*c1^3 + 1/2*c1^2 + c1 + 1
    """
    return rank + capped_logg(total_chern(rank), dim if dim != None else rank)

@cached_function
def kummer(n, cached=True):
    """
    Compute the Chern numbers of the generalized Kummer variety of dimension
    2n. Return a `CobordismClass` object.

    EXAMPLES::

        sage: from bott import kummer
        sage: X = kummer(3)
        sage: print(X.chern_numbers())
        {[6]: 448, [4, 2]: 6784, [2, 2, 2]: 30208}

    TESTS::

        sage: from bott import kummer
        sage: kummer(0)
        Cobordism Class of dim 0
        sage: kummer(2) == kummer(2, cached=False)
        True
    """
    if 0 < n <= 13 and cached:
        X = HKCobordismClass(2*n, Cached.Kumn[n][0])
        X.fujiki_constants = MethodType(lambda self: Cached.Kumn[n][1], X)
    else:
        P2n = [hilb_P2(k+1).cobordism_class() for k in range(n+1)]
        g = universal_genus(2*(n+1), twist=1)
        R = PolynomialRing(g.parent(), 1, "z")
        z = R.gen()
        g = by_degree(g, 2*(n+1))
        K = QQ((2*(n+1)**2, 9)) * by_degree(by_degree(capped_log(R(1)+sum(P2n[k].integral(g[2*(k+1)]) * z**(k+1) for k in range(0, n+1)), n+1), n+1)[n+1].coefficients()[0].coefficients()[0], 2*n)[2*n]
        PP = [Pn(k+1).cobordism_class() for k in range(2*n)]
        X = HKCobordismClass(2*n, compute_sum(K, 2*n, PP))
    X.deformation_type = "kummer"
    return X

def OG(n):
    """
    Return the cobordism class of O'Grady examples OG6 and OG10.

    EXAMPLES::

        sage: from bott import OG
        sage: OG(6)
        Cobordism Class of dim 6
    """
    if n == 6:
        X = HKCobordismClass(6, {Partition(p): QQ(v) for (p, v) in [
            ([6], 1920),
            ([4,2], 7680),
            ([2,2,2], 30720)
        ]})
        X.deformation_type = "OG6"
        return X
    elif n == 10:
        X = HKCobordismClass(10, {Partition(p): QQ(v) for (p, v) in [
            ([10], 176904),
            ([8,2], 1791720),
            ([6,4], 5159700),
            ([6,2,2], 12383280),
            ([4,4,2], 22113000),
            ([4,2,2,2], 53071200),
            ([2,2,2,2,2], 127370880)
        ]})
        X.deformation_type = "OG10"
        return X
    else:
        raise ValueError("Use OG(6) and OG(10) for O'Grady examples")

def to_P2k(L, k):
    """
    Return the tautological vector bundle L^[k] on P2^[k].

    TESTS::

        sage: from bott import Pn, to_P2k
        sage: L = O(Pn(2, w=[0,1,10])); L2 = to_P2k(L, 2); X = L2.parent
        sage: (L2 * X.T.dual()).chi()
        -2
    """
    P = L.parent
    w = P.w
    P2k = hilb_P2(k, w=w)
    def loc(p):
        ws = []
        for k in GF(3):
            for (j, a) in enumerate(p[k]):
                for i in range(a):
                    for l in L.weight(P.points[k]):
                        ws.append(l + i*(w[k]-w[k+1]) + j*(w[k]-w[k+2]))
        return ws
    return TnBundle(P2k, k*rank(L), loc)

def to_P1xP1k(L, k):
    """
    Return the tautological vector bundle L^[k] on P1xP1^[k].

    TESTS::

        sage: from bott import Pn, to_P1xP1k
        sage: PP = Pn(1, w=[0,1]) * Pn(1, w=[0,2]);
        sage: L = O(PP); L2 = to_P1xP1k(L, 2); X = L2.parent
        sage: (L2 * X.T.dual()).chi()
        -3
    """
    P = L.parent
    w = P.w
    P1xP1k = hilb_P1xP1(k, w=w)
    def loc(p):
        ws = []
        for k0 in GF(2):
            for k1 in GF(2):
                n = 2*int(k0) + int(k1)
                for (j, a) in enumerate(p[n]):
                    for i in range(a):
                        for l in L.weight(P.points[n]):
                            ws.append(l + i*(w[0][k0]-w[0][k0+1]) + j*(w[1][k1]-w[1][k1+1]))
        return ws
    return TnBundle(P1xP1k, k*rank(L), loc)

def riemann_roch_polynomial(X, alpha=1, base_ring=QQ):
    """
    Return the Riemann-Roch polynomial of X, using the Fujiki constants of the
    Todd class.

    EXAMPLES::

        sage: from bott import hilb_K3, riemann_roch_polynomial
        sage: riemann_roch_polynomial(hilb_K3(2))
        1/8*q^2 + 5/4*q + 3
        sage: factor(_)
        (1/8) * (q + 4) * (q + 6)

    One can use the argument `alpha` to use other powers of the Todd class,
    e.g., the square root of the Todd class.
    
    EXAMPLES::

        sage: riemann_roch_polynomial(hilb_K3(2), 1/2)
        1/8*q^2 + 5/8*q + 25/32
        sage: factor(_)
        (1/8) * (q + 5/2)^2
    """
    n = dim(X)//2
    td = capped_exp(alpha*capped_log(todd(2*n, base_ring), 2*n), 2*n).homogeneous_components()
    R = PolynomialRing(base_ring, "q")
    return R([X.fujiki_constant(td[2*n-2*i])/factorial(2*i) for i in range(n+1)])

class CharClassPolyHK:
    """
    The class of a universal Characteristic class polynomial, with only even
    degree terms.

    Four classes are available:

        - `c`:  Chern class
        - `ch`: Chern character
        - `td`: Todd class
        - `sq`: square root of Todd class
    
    EXAMPLES::

        sage: from bott import c, ch, td, sq
        sage: td
        Characteristic class polynomial with only even degree terms
        sage: td[4]
        1/240*c2^2 - 1/720*c4
        sage: (sq*sq)[4]
        1/240*c2^2 - 1/720*c4

    Note that ch_0 has no well-defined value since it depends on the dimension.

    EXAMPLES::

        sage: ch[0]
        Traceback (most recent call last):
        ...
        ValueError: ch_0 has no well-defined value
    """
    def __init__(self, f):
        self._f = f
    def __getitem__(self, n):
        x = self._f(n).homogeneous_components()[n]
        c = parent(x).gens()
        return parent(x)(x([c[i] if i%2 == 1 else 0 for i in range(n)]))
    def __mul__(self, other):
        if self == ch or other == ch:
            raise ValueError("cannot multiply with ch, as ch_0 is not well-defined")
        else:
            return CharClassPolyHK(lambda n: self._f(n) * other._f(n))
    def __repr__(self):
        return "Characteristic class polynomial with only even degree terms"

c  = CharClassPolyHK(total_chern)
td = CharClassPolyHK(todd)
sq = CharClassPolyHK(sqrt_todd)
def _ch(n):
    if n == 0:
        raise ValueError("ch_0 has no well-defined value")
    else:
        return capped_logg(total_chern(n), n)
ch = CharClassPolyHK(_ch)
