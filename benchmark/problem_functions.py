__author__ = 'belkhir'

import numpy as np
from numpy import inf, array, dot, exp, log, sqrt, sum, isscalar, isfinite


class FitnessFunctions(object):
    """ versatile container for test objective functions """

    def __init__(self):
        self.counter = 0  # number of calls or any other practical use

    def rot(self, x, fun, rot=1, args=()):
        """returns ``fun(rotation(x), *args)``, ie. `fun` applied to a rotated argument"""
        if len(np.shape(array(x))) > 1:  # parallelized
            res = []
            for x in x:
                res.append(self.rot(x, fun, rot, args))
            return res

        if rot:
            return fun(rotate(x, *args))
        else:
            return fun(x)

    def somenan(self, x, fun, p=0.1):
        """returns sometimes np.NaN, otherwise fun(x)"""
        if np.random.rand(1) < p:
            return np.NaN
        else:
            return fun(x)

    def rand(self, x):
        """Random test objective function"""
        return np.random.random(1)[0]

    def linear(self, x):
        return -x[0]

    def lineard(self, x):
        if 1 < 3 and any(array(x) < 0):
            return np.nan
        if 1 < 3 and sum([(10 + i) * x[i] for i in rglen(x)]) > 50e3:
            return np.nan
        return -sum(x)

    def sphere(self, x):
        """Sphere (squared norm) test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        return sum((x + 0) ** 2)

    def grad_sphere(self, x, *args):
        return 2 * array(x, copy=False)

    def grad_to_one(self, x, *args):
        return array(x, copy=False) - 1

    def sphere_pos(self, x):
        """Sphere (squared norm) test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        c = 0.0
        if x[0] < c:
            return np.nan
        return -c ** 2 + sum((x + 0) ** 2)

    def spherewithoneconstraint(self, x):
        return sum((x + 0) ** 2) if x[0] > 1 else np.nan

    def elliwithoneconstraint(self, x, idx=[-1]):
        return self.ellirot(x) if all(array(x)[idx] > 1) else np.nan

    def spherewithnconstraints(self, x):
        return sum((x + 0) ** 2) if all(array(x) > 1) else np.nan

    # zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    def noisysphere(self, x, noise=2.10e-9, cond=1.0, noise_offset=0.10):
        """noise=10 does not work with default popsize, noise handling does not help """
        return self.elli(x, cond=cond) * (1 + noise * np.random.randn() / len(x)) + noise_offset * np.random.rand()

    def spherew(self, x):
        """Sphere (squared norm) with sum x_i = 1 test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        # s = sum(abs(x))
        # return sum((x/s+0)**2) - 1/len(x)
        # return sum((x/s)**2) - 1/len(x)
        return -0.01 * x[0] + abs(x[0]) ** -2 * sum(x[1:] ** 2)

    def partsphere(self, x):
        """Sphere (squared norm) test objective function"""
        self.counter += 1
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        dim = len(x)
        x = array([x[i % dim] for i in xrange(2 * dim)])
        N = 8
        i = self.counter % dim
        # f = sum(x[i:i + N]**2)
        f = sum(x[np.random.randint(dim, size=N)] ** 2)
        return f

    def sectorsphere(self, x):
        """asymmetric Sphere (squared norm) test objective function"""
        return sum(x ** 2) + (1e6 - 1) * sum(x[x < 0] ** 2)

    def cornersphere(self, x):
        """Sphere (squared norm) test objective function constraint to the corner"""
        nconstr = len(x) - 0
        if any(x[:nconstr] < 1):
            return np.NaN
        return sum(x ** 2) - nconstr

    def cornerelli(self, x):
        """ """
        if any(x < 1):
            return np.NaN
        return self.elli(x) - self.elli(np.ones(len(x)))

    def cornerellirot(self, x):
        """ """
        if any(x < 1):
            return np.NaN
        return self.ellirot(x)

    def normalSkew(self, f):
        N = np.random.randn(1)[0] ** 2
        if N < 1:
            N = f * N  # diminish blow up lower part
        return N

    def noiseC(self, x, func=sphere, fac=10, expon=0.8):
        f = func(self, x)
        N = np.random.randn(1)[0] / np.random.randn(1)[0]
        return max(1e-19, f + (float(fac) / len(x)) * f ** expon * N)

    def noise(self, x, func=sphere, fac=10, expon=1):
        f = func(self, x)
        # R = np.random.randn(1)[0]
        R = np.log10(f) + expon * abs(10 - np.log10(f)) * np.random.rand(1)[0]
        # sig = float(fac)/float(len(x))
        # R = log(f) + 0.5*log(f) * random.randn(1)[0]
        # return max(1e-19, f + sig * (f**np.log10(f)) * np.exp(R))
        # return max(1e-19, f * np.exp(sig * N / f**expon))
        # return max(1e-19, f * normalSkew(f**expon)**sig)
        return f + 10 ** R  # == f + f**(1+0.5*RN)

    def cigar(self, x, rot=0, cond=1e6, noise=0):
        """Cigar test objective function"""
        if rot:
            x = rotate(x)
        x = [x] if isscalar(x[0]) else x  # scalar into list
        f = [(x[0] ** 2 + cond * sum(x[1:] ** 2)) * np.exp(noise * np.random.randn(1)[0] / len(x)) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar

    def grad_cigar(self, x, *args):
        grad = 2 * 1e6 * np.array(x)
        grad[0] /= 1e6
        return grad

    def diagonal_cigar(self, x, cond=1e6):
        axis = np.ones(len(x)) / len(x) ** 0.5
        proj = dot(axis, x) * axis
        s = sum(proj ** 2)
        s += cond * sum((x - proj) ** 2)
        return s

    def tablet(self, x, rot=0):
        """Tablet test objective function"""
        if rot and rot is not fcts.tablet:
            x = rotate(x)
        x = [x] if isscalar(x[0]) else x  # scalar into list
        f = [1e6 * x[0] ** 2 + sum(x[1:] ** 2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar

    def grad_tablet(self, x, *args):
        grad = 2 * np.array(x)
        grad[0] *= 1e6
        return grad

    def cigtab(self, y):
        """Cigtab test objective function"""
        X = [y] if isscalar(y[0]) else y
        f = [1e-4 * x[0] ** 2 + 1e4 * x[1] ** 2 + sum(x[2:] ** 2) for x in X]
        return f if len(f) > 1 else f[0]

    def twoaxes(self, y):
        """Cigtab test objective function"""
        X = [y] if isscalar(y[0]) else y
        N2 = len(X[0]) // 2
        f = [1e6 * sum(x[0:N2] ** 2) + sum(x[N2:] ** 2) for x in X]
        return f if len(f) > 1 else f[0]

    def ellirot(self, x):
        return fcts.elli(array(x), 1)

    def hyperelli(self, x):
        N = len(x)
        return sum((np.arange(1, N + 1) * x) ** 2)

    def halfelli(self, x):
        l = len(x) // 2
        felli = self.elli(x[:l])
        return felli + 1e-8 * sum(x[l:] ** 2)

    def elli(self, x, rot=0, xoffset=0, cond=1e6, actuator_noise=0.0, both=False):
        """Ellipsoid test objective function"""
        if not isscalar(x[0]):  # parallel evaluation
            return [self.elli(xi, rot) for xi in x]  # could save 20% overall
        if rot:
            x = rotate(x)
        N = len(x)
        if actuator_noise:
            x = x + actuator_noise * np.random.randn(N)

        ftrue = sum(cond ** (np.arange(N) / (N - 1.)) * (x + xoffset) ** 2)

        alpha = 0.49 + 1. / N
        beta = 1
        felli = np.random.rand(1)[0] ** beta * ftrue * \
                max(1, (10. ** 9 / (ftrue + 1e-99)) ** (alpha * np.random.rand(1)[0]))
        # felli = ftrue + 1*np.random.randn(1)[0] / (1e-30 +
        # np.abs(np.random.randn(1)[0]))**0
        if both:
            return (felli, ftrue)
        else:
            # return felli  # possibly noisy value
            return ftrue  # + np.random.randn()

    def grad_elli(self, x, *args):
        cond = 1e6
        N = len(x)
        return 2 * cond ** (np.arange(N) / (N - 1.)) * array(x, copy=False)

    def fun_as_arg(self, x, *args):
        """``fun_as_arg(x, fun, *more_args)`` calls ``fun(x, *more_args)``.

        Use case::

            fmin(cma.fun_as_arg, args=(fun,), gradf=grad_numerical)

        calls fun_as_args(x, args) and grad_numerical(x, fun, args=args)

        """
        fun = args[0]
        more_args = args[1:] if len(args) > 1 else ()
        return fun(x, *more_args)

    def grad_numerical(self, x, func, epsilon=None):
        """symmetric gradient"""
        eps = 1e-8 * (1 + abs(x)) if epsilon is None else epsilon
        grad = np.zeros(len(x))
        ei = np.zeros(len(x))  # float is 1.6 times faster than int
        for i in rglen(x):
            ei[i] = eps[i]
            grad[i] = (func(x + ei) - func(x - ei)) / (2 * eps[i])
            ei[i] = 0
        return grad

    def elliconstraint(self, x, cfac=1e8, tough=True, cond=1e6):
        """ellipsoid test objective function with "constraints" """
        N = len(x)
        f = sum(cond ** (np.arange(N)[-1::-1] / (N - 1)) * x ** 2)
        cvals = (x[0] + 1,
                 x[0] + 1 + 100 * x[1],
                 x[0] + 1 - 100 * x[1])
        if tough:
            f += cfac * sum(max(0, c) for c in cvals)
        else:
            f += cfac * sum(max(0, c + 1e-3) ** 2 for c in cvals)
        return f

    def rosen(self, x, alpha=1e2):
        """Rosenbrock test objective function"""
        x = [x] if isscalar(x[0]) else x  # scalar into list
        f = [sum(alpha * (x[:-1] ** 2 - x[1:]) ** 2 + (1. - x[:-1]) ** 2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar

    def grad_rosen(self, x, *args):
        N = len(x)
        grad = np.zeros(N)
        grad[0] = 2 * (x[0] - 1) + 200 * (x[1] - x[0] ** 2) * -2 * x[0]
        i = np.arange(1, N - 1)
        grad[i] = 2 * (x[i] - 1) - 400 * (x[i + 1] - x[i] ** 2) * x[i] + 200 * (x[i] - x[i - 1] ** 2)
        grad[N - 1] = 200 * (x[N - 1] - x[N - 2] ** 2)
        return grad

    def diffpow(self, x, rot=0):
        """Diffpow test objective function"""
        N = len(x)
        if rot:
            x = rotate(x)
        return sum(np.abs(x) ** (2. + 4. * np.arange(N) / (N - 1.))) ** 0.5

    def rosenelli(self, x):
        N = len(x)
        return self.rosen(x[:N / 2]) + self.elli(x[N / 2:], cond=1)

    def ridge(self, x, expo=2):
        x = [x] if isscalar(x[0]) else x  # scalar into list
        f = [x[0] + 100 * np.sum(x[1:] ** 2) ** (expo / 2.) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar

    def ridgecircle(self, x, expo=0.5):
        """happy cat by HG Beyer"""
        a = len(x)
        s = sum(x ** 2)
        return ((s - a) ** 2) ** (expo / 2) + s / a + sum(x) / a

    def happycat(self, x, alpha=1. / 8):
        s = sum(x ** 2)
        return ((s - len(x)) ** 2) ** alpha + (s / 2 + sum(x)) / len(x) + 0.5

    def flat(self, x):
        return 1
        return 1 if np.random.rand(1) < 0.9 else 1.1
        return np.random.randint(1, 30)

    def branin(self, x):
        # in [0,15]**2
        y = x[1]
        x = x[0] + 5
        return (y - 5.1 * x ** 2 / 4 / np.pi ** 2 + 5 * x / np.pi - 6) ** 2 + 10 * (1 - 1 / 8 / np.pi) * np.cos(
            x) + 10 - 0.397887357729738160000

    def goldsteinprice(self, x):
        x1 = x[0]
        x2 = x[1]
        return (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) * (
            30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)) - 3

    def griewank(self, x):
        # was in [-600 600]
        x = (600. / 5) * x
        return 1 - np.prod(np.cos(x / sqrt(1. + np.arange(len(x))))) + sum(x ** 2) / 4e3

    def rastrigin(self, x):
        """Rastrigin test objective function"""
        if not isscalar(x[0]):
            N = len(x[0])
            return [10 * N + sum(xi ** 2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]
            # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
        N = len(x)
        return 10 * N + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    def schaffer(self, x):
        """ Schaffer function x0 in [-100..100]"""
        N = len(x)
        s = x[0:N - 1] ** 2 + x[1:N] ** 2
        return sum(s ** 0.25 * (np.sin(50 * s ** 0.1) ** 2 + 1))

    def schwefelelli(self, x):
        s = 0
        f = 0
        for i in rglen(x):
            s += x[i]
            f += s ** 2
        return f

    def schwefelmult(self, x, pen_fac=1e4):
        """multimodal Schwefel function with domain -500..500"""
        y = [x] if isscalar(x[0]) else x
        N = len(y[0])
        f = array([418.9829 * N - 1.27275661e-5 * N - sum(x * np.sin(np.abs(x) ** 0.5))
                   + pen_fac * sum((abs(x) > 500) * (abs(x) - 500) ** 2) for x in y])
        return f if len(f) > 1 else f[0]

    def optprob(self, x):
        n = np.arange(len(x)) + 1
        f = n * x * (1 - x) ** (n - 1)
        return sum(1 - f)

    def lincon(self, x, theta=0.01):
        """ridge like linear function with one linear constraint"""
        if x[0] < 0:
            return np.NaN
        return theta * x[1] + x[0]

    def rosen_nesterov(self, x, rho=100):
        """needs exponential number of steps in a non-increasing f-sequence.

        x_0 = (-1,1,...,1)
        See Jarre (2011) "On Nesterov's Smooth Chebyshev-Rosenbrock Function"

        """
        f = 0.25 * (x[0] - 1) ** 2
        f += rho * sum((x[1:] - 2 * x[:-1] ** 2 + 1) ** 2)
        return f

    def powel_singular(self, x):
        # ((8 * np.sin(7 * (x[i] - 0.9)**2)**2 ) + (6 * np.sin()))
        res = np.sum((x[i - 1] + 10 * x[i]) ** 2 + 5 * (x[i + 1] - x[i + 2]) ** 2 +
                     (x[i] - 2 * x[i + 1]) ** 4 + 10 * (x[i - 1] - x[i + 2]) ** 4
                     for i in xrange(1, len(x) - 2))
        return 1 + res

    def styblinski_tang(self, x):
        """in [-5, 5]
        """
        # x_opt = N * [-2.90353402], seems to have essentially
        # (only) 2**N local optima
        return (39.1661657037714171054273576010019 * len(x)) ** 1 + \
               sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2

    def trid(self, x):
        return sum((x - 1) ** 2) - sum(x[:-1] * x[1:])

    def bukin(self, x):
        """Bukin function from Wikipedia, generalized simplistically from 2-D.

        http://en.wikipedia.org/wiki/Test_functions_for_optimization"""
        s = 0
        for k in xrange((1 + len(x)) // 2):
            z = x[2 * k]
            y = x[min((2 * k + 1, len(x) - 1))]
            s += 100 * np.abs(y - 0.01 * z ** 2) ** 0.5 + 0.01 * np.abs(z + 10)
        return s


fcts = FitnessFunctions()
Fcts = fcts  # for cross compatibility, as if the functions were static members of class Fcts
FF = fcts

# x = np.random.rand(2)
# print(x)
# y = FF.sphere(x)
# print(y)
