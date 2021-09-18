from sage.all import *

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
                if c.parent().ngens() == 1:
                    d = [d]
                pp = []
                k = 0
                for F in Fs:
                    pp.append(Partition([i+1 for i in range(F.rank-1,-1,-1) for j in range(d[k+i])]))
                    k += F.rank
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
        self.rank = r
        self.weight = weight

    def integral(self, c=None):
        X = self.parent
        if c == None: # top Chern class
            p = Partition([X.dim])
            return self.integrals([p])[p]
        if isinstance(c, Partition): # Chern class as Partition
            return self.integrals([c])[c]
        if isinstance(c, list): # Chern class as list
            c.sort(reverse=True)
            c = Partition(c)
            return self.integrals([c])[c]
        # Chern class as polynomial
        return self.parent.integrals([self], c)

    def integrals(self, partitions):
        ans = self.parent._TnVariety__integrals([self], [(p,) for p in partitions])
        return {p: ans[(p,)] for p in partitions}

    def dual(self):
        return TnBundle(self.parent, self.rank, lambda p: [-wi for wi in self.weight(p)])

    def det(self):
        return TnBundle(self.parent, 1, lambda p: [sum(self.weight(p))])

    def symmetric_power(self, k):
        return TnBundle(self.parent, binomial(self.rank+k-1,k), lambda p: [sum(c) for c in sym(self.weight(p), k)])

    def exterior_power(self, k):
        return TnBundle(self.parent, binomial(self.rank, k), lambda p: [sum(c) for c in comb(self.weight(p), k)])

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
        R = PolynomialRing(QQ, ["c"+str(i+1) for i in range(self.rank)] + ["d"+str(i+1) for i in range(X.dim)], order=TermOrder("wdeglex", list(range(1, self.rank+1)) + list(range(1, X.dim+1))))
        ch = self.rank + capped_logg(sum(R.gens()[:self.rank]), X.dim)
        td = todd(X.dim)
        td = hom(parent(td), R, R.gens()[self.rank:])(td)
        return X.integrals([self, X.T], ch * td) # HRR

    def __check_other(self, other):
        if not isinstance(other, TnBundle):
            raise TypeError(str(other) + " is not a TnBundle")
        if not other.parent == self.parent:
            raise ValueError("Parents do not agree")

    def __add__(self, other):
        self.__check_other(other)
        return TnBundle(self.parent, self.rank + other.rank, lambda p: self.weight(p) + other.weight(p))

    def __mul__(self, other):
        self.__check_other(other)
        return TnBundle(self.parent, self.rank * other.rank, lambda p: [u + v for u in self.weight(p) for v in other.weight(p)])

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
        return TnBundle(self.parent * other.parent, self.rank * other.rank, lambda p: [u + v for u in self.weight(p[0]) for v in other.weight(p[1])])

    def __repr__(self):
        return "TnBundle of rank " + str(self.rank) + " on TnVariety of dim " + str(self.parent.dim)

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
    """
    def __init__(self, n, chern_numbers, check=True):
        self.dim = n
        self.__cn = chern_numbers
        if check and not all([p.size() == n for p in chern_numbers.keys()]):
            raise ValueError(str(p) + " is not a partition of " + str(n))
        for p in Partitions(n):
            if not p in self.__cn.keys():
                self.__cn[p] = 0

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
        if type(p) == list:
            p.sort(reverse=True)
            p = Partition(p)
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
                if c.parent().ngens() == 1:
                    d = [d]
                p = Partition([i+1 for i in range(self.dim-1,-1,-1) for j in range(d[i])])
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
        if type(p) == list:
            p.sort(reverse=True)
            p = Partition(p)
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

    def __repr__(self):
        return "Cobordism Class of dim " + str(self.dim)

@cached_function
def _product(m, n):
    R = PolynomialRing(QQ, ["c"+str(i) for i in range(m)] + ["d"+str(i) for i in range(n)], order=TermOrder("wdeglex", list(range(1,m+1)) + list(range(1,n+1))))
    TX = 1+sum(R.gens()[0:m])
    TY = 1+sum(R.gens()[m:m+n])
    cTXY = TX * TY
    c = by_degree(cTXY, m+n)
    pp = Partitions(m+n)
    ans = {p: {} for p in pp}
    for p in pp:
        cp = prod(c[i] for i in p)
        for monom in cp.monomials():
            d = monom.exponents()[0]
            if sum(d[i] * (i+1) for i in range(m)) == m:
                p1 = Partition([i+1 for i in range(m-1,-1,-1) for j in range(d[i])])
                p2 = Partition([i+1 for i in range(n-1,-1,-1) for j in range(d[i+m])])
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
    if k < 0 or k > n:
        raise ValueError("wrong input for Grassmannian")
    if w == None:
        w = list(range(n))
    G = TnVariety(k*(n-k), [tuple(c) for c in Combinations(n, k)])
    U = TnBundle(G, k, lambda p: [w[i] for i in p])
    Q = TnBundle(G, n-k, lambda p: [w[i] for i in range(n) if not i in p])
    G.T = U.dual() * Q
    G.O1 = Q.det()
    G.bundles = [U, Q]
    G.w = w
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
    ranks = [dims[0]] + [dims[i+1] - dims[i] for i in range(l-1)]
    d = sum(ranks[i] * (n - dims[i]) for i in range(l))
    def enum(i, rest):
        if i == l:
            return [[rest]]
        return [[x] + y for x in Combinations(rest, ranks[i-1]) for y in enum(i+1, [r for r in rest if not r in x])]
    Fl = TnVariety(d, [tuple(c) for c in enum(1, range(n))])
    def closure(i):
        return lambda p: [w[j] for j in p[i]]
    Fl.bundles = [TnBundle(Fl, ranks[i], closure(i)) for i in range(l)]
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
    if x.parent().ngens() == 1:
        g = x.parent().gen()
        l = x.list()
        l += [0] * (n+1-len(l))
        for i in range(max(n, len(l))):
            c[i] = l[i] * g**i
        return c
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

def capped_sqrt(x, n):
    return capped_exp(capped_log(x, n)/2, n)

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

@cached_function
def hilb_K3(n):
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
    """
    return hilb_surface(n, 0, 24)

def hilb_surface(n, c1c1, c2, parent=QQ):
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
        sage: X = hilb_surface(2, a, b, parent=SR)
        sage: print(X.chern_numbers())
        {[4]: 1/2*b^2 + 3/2*b, [3, 1]: a*b + 3*a, [2, 2]: 1/2*a^2 + b^2 + a + 21/2*b, [2, 1, 1]: a^2 + a*b + 6*a, [1, 1, 1, 1]: 3*a^2}
    """
    a, b = matrix([[9,8],[3,4]]).solve_right(vector([c1c1, c2]))
    S = PolynomialRing(parent, ["a"+str(i+1) for i in range(n)]+["b"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple(range(1,n+1))+tuple(range(1,n+1))))
    A, B = gens(S)[:n], gens(S)[n:]
    HS = capped_exp(a*capped_log(S(1)+sum(A), n)+b*capped_log(S(1)+sum(B), n), n)
    P2n    = [hilb_P2(k+1).cobordism_class()    for k in range(n)]
    P1xP1n = [hilb_P1xP1(k+1).cobordism_class() for k in range(n)]
    ans = compute_sum(HS, 2*n, P2n+P1xP1n, check=lambda d: d == n)
    # in case one works in the Symbolic Ring
    if parent == SR:
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

def universal_genus(n, images=None, twist=0):
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

        sage: universal_genus(3, [QQ(1)]*3)
        1/24*c1*c2 + 1/12*c1^2 + 1/12*c2 + 1/2*c1 + 1
    """
    if images == None:
        S = PolynomialRing(QQ, ["P"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple(range(1,n+1))))
        images = [S(1)] + list(S.gens())
    else:
        S = images[0].parent()
        images = [S(1)] + images
    R = PolynomialRing(S, ["c"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple(range(1,n+1))))
    g = genus(capped_logg(R(sum(R.gens())), n), _taylor(images, n), n, twist=twist)
    return g

def todd(n):
    """
    Compute the Todd genus as a polynomial in Chern classes.

    EXAMPLES::

        sage: from bott import todd
        sage: todd(3)
        1/24*c1*c2 + 1/12*c1^2 + 1/12*c2 + 1/2*c1 + 1
    """
    return universal_genus(n, [QQ(1)]*n)

def sqrt_todd(n):
    """
    Compute the square root of the Todd genus as a polynomial in Chern
    classes.

    EXAMPLES::

        sage: from bott import hilb_K3, sqrt_todd
        sage: sqrt_todd(2)
        1/96*c1^2 + 1/24*c2 + 1/4*c1 + 1
        sage: hilb_K3(2).integral(sqrt_todd(4))
        25/32
    """
    return capped_sqrt(todd(n), n)

def chern_character(rank, dim=None):
    """
    Compute the Chern character as a polynomial in Chern classes.

    EXAMPLES::

        sage: from bott import chern_character
        sage: chern_character(3)
        1/6*c1^3 - 1/2*c1*c2 + 1/2*c3 + 1/2*c1^2 - c2 + c1 + 3
        sage: chern_character(1, 4)
        1/24*c1^4 + 1/6*c1^3 + 1/2*c1^2 + c1 + 1
    """
    R = PolynomialRing(QQ, ["c"+str(i+1) for i in range(rank)], order=TermOrder("wdeglex", tuple(range(1,rank+1))))
    return rank + capped_logg(R(sum(R.gens())), dim if dim != None else rank)

@cached_function
def kummer(n):
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
    """
    P2n = [hilb_P2(k+1).cobordism_class() for k in range(n+1)]
    g = universal_genus(2*(n+1), twist=1)
    z = PowerSeriesRing(g.parent(), "z", default_prec=n+2).gen()
    g = by_degree(g, 2*(n+1))
    K = QQ((2*(n+1)**2, 9)) * by_degree(by_degree(log(1+sum(P2n[k].integral(g[2*(k+1)]) * z**(k+1) for k in range(0, n+1))), n+1)[n+1].coefficients()[0].coefficients()[0], 2*n)[2*n]
    PP = [Pn(k+1).cobordism_class() for k in range(2*n)]
    return CobordismClass(2*n, compute_sum(K, 2*n, PP))

def OG(n):
    """
    Return the cobordism class of O'Grady examples OG6 and OG10.

    EXAMPLES::

        sage: from bott import OG
        sage: OG(6)
        Cobordism Class of dim 6
    """
    if n == 6:
        return CobordismClass(6, {Partition([6]): 1920, Partition([4,2]): 7680, Partition([2,2,2]): 30720})
    elif n == 10:
        return CobordismClass(10, {Partition([10]): 176904, Partition([8,2]): 1791720, Partition([6,4]): 5159700, Partition([6,2,2]): 12383280, Partition([4,4,2]): 22113000, Partition([4,2,2,2]): 53071200, Partition([2,2,2,2,2]): 127370880})
    else:
        raise ValueError("Use OG(6) and OG(10) for O'Grady examples")

def to_P2k(L, k):
    P = L.parent
    w = P.w
    P2k = hilb_P2(k, w=w)
    def loc(p):
        ws = []
        for k in GF(3):
            for (j, a) in enumerate(p[k]):
                for i in range(a):
                    for l in L.weight(P.points[k]):
                        ws.append(l + i*(w[k+1]-w[k]) + j*(w[k+2]-w[k]))
        return ws
    return TnBundle(P2k, k*L.rank, loc)

def to_P1xP1k(L, k):
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
                            ws.append(l + i*(w[0][k0+1]-w[0][k0]) + j*(w[1][k1+1]-w[1][k1]))
        return ws
    return TnBundle(P1xP1k, k*L.rank, loc)

def fujiki_constant(n):
    """
    Return the Fujiki constants of the Chern classes on K3[n].

    EXAMPLES::

        sage: from bott import fujiki_constant
        sage: print(fujiki_constant(2))
        {[]: 3, [2]: 30, [4]: 324, [2, 2]: 828}
    """
    g = universal_genus(2*n)
    Omega = parent(g).base_ring()
    P = [Omega(0)] + list(Omega.gens())
    Rt = PolynomialRing(Omega, "t")
    t = Rt.gen()
    A, B = Rt(1), Rt(1)
    for k in range(1, n+1):
        R = PolynomialRing(Omega, ["c"+str(i+1) for i in range(2*k)] + ["E"], order=TermOrder("wdeglex", list(range(1, 2*k+1)) + [1]))
        gk = universal_genus(2*k)
        c = hom(parent(gk), R, R.gens()[:-1])(gk) * capped_exp(R.gens()[-1], 2*k)
        # pick a random weight w
        Ok = to_P2k(Pn(2, w=[0,12,1321]).O(), k).det()
        P2k = Ok.parent
        A += t**k * P2k.integrals([P2k.T, Ok], c)
        # pick random weights w
        Ok = to_P1xP1k((Pn(1, w=[0,121]) * Pn(1, w=[0,213])).O(), k)
        P1xP1k = Ok.parent
        B += t**k * P1xP1k.integrals([P1xP1k.T, Ok], c)
    C = capped_exp(-16 * capped_log(A, n) + 18 * capped_log(B, n), n)
    c = [0] + list(parent(g).gens())
    ans = {}
    coeffs_in_Pn = lambda x, m: [x.monomial_coefficient(Omega(prod(P[k] for k in p))) for p in Partitions(m)]
    for m in range(0, n+1):
        M = matrix([coeffs_in_Pn(g.monomial_coefficient(g.parent(prod(c[2*k] for k in p))), 2*m) for p in Partitions(m)])
        Cn = by_degree(C.monomial_coefficient(t**n), 2*n)
        coeffs = M.solve_left(vector(coeffs_in_Pn(Cn[2*m], 2*m)))
        for (q, v) in zip(Partitions(m), coeffs):
            q = Partition([2*qi for qi in q])
            if m == 0 and n == 1:
                ans[q] = 1
            else:
                ans[q] = v * factorial(2*(n-m)) / (2-2*n)**(n-m)
    return ans
