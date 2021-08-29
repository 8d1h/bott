# from functools import reduce
# from itertools import product

# from sage.symbolic.all import SR
# from sage.rings.all import QQ
# from sage.rings.polynomial.all import PolynomialRing, TermOrder
# from sage.matrix.all import matrix
# from sage.modules.all import vector
# from sage.combinat.all import Combinations, Partition, Partitions
# from sage.misc.all import gens, cached_function, prod
# from sage.parallel.all import parallel

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
    def __init__(self, n, points, weight):
        self.dim = n
        self.__points = points
        self.__weight = weight

    def fixed_points(self):
        """
        Return the list of fixed points of the torus action.
        """
        return list(self.__points)

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
        @parallel(Nthreads.n)
        def compute(points):
            ans = {p: 0 for p in partitions}
            for pt in points:
                w = self.__weight(pt)
                cherns = [1] + [chern(k, w) for k in range(1, len(w)+1)]
                ctop = cherns[-1]
                for p in ans.keys():
                    ans[p] += prod(cherns[k] for k in p) * QQ((1, ctop))
            return ans

        partitions = Partitions(self.dim)
        each = max(len(self.__points) // Nthreads.n, 10)
        num = len(self.__points) // each
        c = compute([self.__points[i*each:(i+1)*each] for i in range(num)] + [self.__points[num*each:]])
        ans = {p: 0 for p in partitions}
        for (_, ci) in c:
            for p in ans.keys():
                ans[p] += ci[p]
        return ans

    def cobordism_class(self):
        """
        Return the cobordism class.

        EXAMPLES::

            sage: from bott import Pn
            sage: Pn(2).cobordism_class()
            Cobordism Class of dim 2
        """
        return CobordismClass(self.dim, self.chern_numbers(), check=False)

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
        points = cartesian_product([self.__points, other.__points])
        weight = lambda p: self.__weight(p[0]) + other.__weight(p[1])
        return TnVariety(n, points, weight)

    def __repr__(self):
        return "TnVariety of dim " + str(self.dim) + " with " + str(len(self.__points)) + " fixed points"

class CobordismClass:
    """
    A cobordism class represented by the dimension and the Chern numbers.
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
            p = Partition(p)
        if p.size() != self.dim:
            raise ValueError(str(p) + " is not a partition of " + str(self.dim))
        return self.__cn[p]

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
        monoms, coeffs = cp.monomials(), cp.coefficients()
        for i in range(len(monoms)):
            d = monoms[i].exponents()[0]
            if sum(d[i] * (i+1) for i in range(m)) == m:
                p1 = Partition([i+1 for i in range(m-1,-1,-1) for j in range(d[i])])
                p2 = Partition([i+1 for i in range(n-1,-1,-1) for j in range(d[i+m])])
                ans[p][(p1,p2)] = coeffs[i]
    return ans

def Pn(n):
    """
    Projective space of dimension n, with a torus action.

    EXAMPLES::

        sage: from bott import Pn
        sage: Pn(3)
        TnVariety of dim 3 with 4 fixed points
    """
    return TnVariety(n, range(n+1), lambda k: [x-k for x in range(0, k)] + [x-k for x in range(k+1, n+1)])

def grassmannian(k, n):
    """
    Grassmannian Gr(k, n), with a torus action.

    EXAMPLES::

        sage: from bott import grassmannian
        sage: grassmannian(2, 4)
        TnVariety of dim 4 with 6 fixed points
    """
    if k <= 0 or k >= n:
        raise ValueError("wrong input for Grassmannian")
    return TnVariety(k*(n-k), Combinations(n, k), lambda v: [b-a for b in range(n) if not b in v for a in v])

def _hilb(n, parts, wt):
    points = [x for pp in parts for x in cartesian_product([Partitions(x) for x in pp])]
    def weight(pt):
        w = []
        for k,l,m in wt:
            if len(pt[k]) > 0:
                b = pt[k].conjugate()
                for s in range(len(pt[k])):
                    j = pt[k][s]
                    for i in range(j):
                        w.append(l*(i-j)   + m*(b[i]-s-1))
                        w.append(l*(j-i-1) + m*(s-b[i]))
        return w
    return TnVariety(2*n, points, weight)

def hilb_P2(n):
    """
    Hilbert scheme of n points on the projective plane, with a torus action.

    EXAMPLES::

        sage: from bott import hilb_P2
        sage: hilb_P2(2)
        TnVariety of dim 4 with 9 fixed points
    """
    return _hilb(n, [[a, b-a-1, n+1-b] for a,b in Combinations(n+2, 2)], [[0,1,n+1], [1,n,-1], [2,-n-1,-n]])

def hilb_P1xP1(n):
    """
    Hilbert scheme of n points on the product of two projective lines, with a
    torus action.

    EXAMPLES::

        sage: from bott import hilb_P1xP1
        sage: hilb_P1xP1(2)
        TnVariety of dim 4 with 14 fixed points
    """
    return _hilb(n, [[a, b-a-1, c-b-1, n+2-c] for a,b,c in Combinations(n+3, 3)], [[0,-n,-1], [1,n,-1], [2,-n,1], [3,n,1]])

def by_degree(x, n):
    c = [0] * (n+1)
    if x.parent().ngens() == 1:
        g = x.parent().gen()
        l = x.list()
        for i in range(max(n, len(l))):
            c[i] = l[i] * g**i
    monoms, coeffs = x.monomials(), x.coefficients()
    for i in range(len(monoms)):
        d = monoms[i].degree() 
        if d <= n:
            if type(coeffs) == dict:
                c[d] += coeffs[monoms[i]] * monoms[i]
            else:
                c[d] += coeffs[i] * monoms[i]
    return c

def capped_log(x, n):
    e = by_degree(x, n)
    p = [e[1]] + [0] * (n-1)
    for i in range(n-1):
        p[i+1] = (i+2) * e[i+2] - sum(e[j+1] * p[i-j] for j in range(i+1))
    return sum(QQ((1, i+1))*p[i] for i in range(n))

def capped_exp(x, n):
    comps = by_degree(x, n)
    p = [i * comps[i] for i in range(n+1)]
    e = [1] + [0] * (n)
    for i in range(n):
        e[i+1] = QQ((1, i+1)) * sum(p[j+1] * e[i-j] for j in range(i+1))
    return sum(e)

def hilb_K3(n):
    """
    Compute the Chern numbers of the Hilbert scheme of n points on a K3
    surface. Return a `CobordismClass` object.

    EXAMPLES::

        sage: from bott import hilb_K3
        sage: X = hilb_K3(3)
        sage: print(X.chern_numbers())
        {[6]: 3200, [4, 2]: 14720, [2, 2, 2]: 36800}
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
    S = PolynomialRing(parent, ["a"+str(i+1) for i in range(n)]+["b"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple([i for i in range(1,n+1)]+[i for i in range(1,n+1)])))
    A, B = gens(S)[:n], gens(S)[n:]
    HS = capped_exp(a*capped_log(1+sum(A), n)+b*capped_log(1+sum(B), n), n)
    monoms, coeffs = HS.monomials(), HS.coefficients()
    ans = {p: 0 for p in Partitions(2*n)}
    P2n    = [hilb_P2(k+1).cobordism_class()    for k in range(n)]
    P1xP1n = [hilb_P1xP1(k+1).cobordism_class() for k in range(n)]
    for i in range(len(monoms)):
        if monoms[i].degree() == n:
            d = monoms[i].exponents()[0]
            c = prod([P2n[k] for k in range(n) for i in range(d[k])] + [P1xP1n[k] for k in range(n) for i in range(d[k+n])])
            for p in ans.keys():
                ans[p] += coeffs[i] * c.chern_number(p)
    # in case one works in the Symbolic Ring
    if parent == SR:
        for p in ans.keys():
            ans[p] = ans[p].expand().simplify()
    return CobordismClass(2*n, ans)
