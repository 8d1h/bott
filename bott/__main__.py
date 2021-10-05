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
        monoms, coeffs = c.monomials(), c.coefficients()
        for i in range(len(monoms)):
            if monoms[i].degree() == self.dim:
                d = monoms[i].exponents()[0]
                p = Partition([i+1 for i in range(self.dim-1,-1,-1) for j in range(d[i])])
                ans += coeffs[i] * self.__cn[p]
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
    c = [x.parent(0)] * (n+1)
    if x.parent().ngens() == 1:
        g = x.parent().gen()
        l = x.list()
        l += [0] * (n+1-len(l))
        for i in range(max(n, len(l))):
            c[i] = l[i] * g**i
        return c
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
    terms = list(zip(x.monomials(), x.coefficients()))
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

def chern_character(n):
    """
    Compute the Chern character as a polynomial in Chern classes.

    EXAMPLES::

        sage: from bott import chern_character
        sage: chern_character(3)
        1/6*c1^3 - 1/2*c1*c2 + 1/2*c3 + 1/2*c1^2 - c2 + c1 + 3
    """
    R = PolynomialRing(QQ, ["c"+str(i+1) for i in range(n)], order=TermOrder("wdeglex", tuple(range(1,n+1))))
    return n + capped_logg(R(sum(R.gens())), n)

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
