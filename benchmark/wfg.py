"""
This module contains the test problems for multiobjective optimization
published by the Walking Fish Group (WFG) [Huband2006]_. The problems here
are a reimplementation of the original C++ code at
http://www.wfg.csse.uwa.edu.au/toolkit/ . They are all minimization problems.
"""

import math
import copy
import random

from benchmark.multiobjective import MultiObjectiveTestProblem
from benchmark.base import BoundConstraintError, Individual

__all__ = ["WFG", "WFGBaseProblem", "WFG1", "WFG2", "WFG3", "WFG4", "WFG5",
           "WFG6", "WFG7", "WFG8", "WFG9"]

def correct_to_01(a, epsilon=1.0e-10):
    """Sets values in [-epsilon, 0] to 0 and in [1, 1 + epsilon] to 1.

    Assumption is that these deviations result from rounding errors.

    """
    assert epsilon >= 0.0
    min_value = 0.0
    max_value = 1.0
    min_epsilon = min_value - epsilon
    max_epsilon = max_value + epsilon
    if a <= min_value and a >= min_epsilon:
        return min_value
    elif a >= max_value and a <= max_epsilon:
        return max_value
    else:
        return a



def vector_in_01(x):
    """Returns True if all elements are in [0, 1]."""
    for element in x:
        if element < 0.0 or element > 1.0:
            return False
    return True



def shape_args_ok(x, m):
    """Helper function."""
    return vector_in_01(x) and m >= 1 and m <= len(x)



def calculate_f(d, x, h, s):
    """Helper function."""
    assert d > 0.0
    assert vector_in_01(x)
    assert vector_in_01(h)
    assert len(x) == len(h)
    assert len(h) == len(s)
    result = []
    for i in range(0, len(h)):
        assert s[i] > 0.0
        result.append(d * x[-1] + s[i] * h[i])
    return result



def b_poly(y, alpha):
    """Transformation function."""
    assert y >= 0.0
    assert y <= 1.0
    assert alpha > 0.0
    assert alpha != 1.0
    return correct_to_01(math.pow(y, alpha))



def b_flat(y, a, b, c):
    """Transformation function."""
    assert y >= 0.0
    assert y <= 1.0
    assert a >= 0.0
    assert a <= 1.0
    assert b >= 0.0
    assert b <= 1.0
    assert c >= 0.0
    assert c <= 1.0
    assert b < c
    assert b != 0.0 or a == 0.0
    assert b != 0.0 or c != 1.0
    assert c != 1.0 or a == 1.0
    assert c != 1.0 or b != 0.0
    tmp1 = min(0.0, math.floor(y - b)) * a * (b - y) / b
    tmp2 = min(0.0, math.floor(c - y)) * (1.0 - a) * (y - c) / (1.0 - c)
    return correct_to_01(a + tmp1 - tmp2)



def b_param(y, u, a, b, c):
    """Transformation function."""
    assert y >= 0.0
    assert y <= 1.0
    assert u >= 0.0
    assert u <= 1.0
    assert a > 0.0
    assert a < 1.0
    assert b > 0.0
    assert b < c
    v = a - (1.0 - 2.0 * u) * abs(math.floor(0.5 - u) + a)
    return correct_to_01(math.pow(y, b + (c - b) * v))



def s_linear(y, a):
    """Transformation function."""
    assert y >= 0.0
    assert y <= 1.0
    assert a > 0.0
    assert a < 1.0
    return correct_to_01(abs(y - a) / abs(math.floor(a - y) + a))



def s_decept(y, a, b, c):
    """Transformation function."""
    assert y >= 0.0
    assert y <= 1.0
    assert a > 0.0
    assert a < 1.0
    assert b > 0.0
    assert b < 1.0
    assert c > 0.0
    assert c < 1.0
    assert a - b > 0.0
    assert a + b < 1.0
    tmp1 = math.floor(y - a + b) * (1.0 - c + (a - b) / b) / (a - b)
    tmp2 = math.floor(a + b - y) * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b)
    return correct_to_01(1.0 + (abs(y - a) - b) * (tmp1 + tmp2 + 1.0 / b))



def s_multi(y, a, b, c):
    """Transformation function."""
    assert y >= 0.0
    assert y <= 1.0
    assert a >= 1
    assert b >= 0.0
    assert (4.0 * a + 2.0) * math.pi >= 4.0 * b
    assert c > 0.0
    assert c < 1.0
    tmp1 = abs(y - c) / (2.0 * (math.floor(c - y) + c))
    tmp2 = (4.0 * a + 2.0) * math.pi * (0.5 - tmp1)
    result = (1.0 + math.cos(tmp2) + 4.0 * b * math.pow(tmp1, 2.0)) / (b + 2.0)
    return correct_to_01(result)



def r_sum(y, w):
    """Transformation function."""
    assert len(y) != 0
    assert len(w) == len(y)
    assert vector_in_01(y)
    numerator = 0.0
    denominator = sum(w)
    for i in range(len(y)):
        assert w[i] > 0.0
        numerator += w[i]*y[i]
    return correct_to_01(numerator / denominator)



def r_nonsep(y, a):
    """Transformation function."""
    assert len(y) != 0
    assert vector_in_01(y)
    assert a >= 1
    assert a <= len(y)
    assert len(y) % a == 0
    numerator = 0.0
    for j in range(len(y)):
        numerator += y[j]
        for k in range(0, a - 1):
            numerator += abs(y[j] - y[(j + k + 1) % len(y)])
    tmp = math.ceil(a / 2.0)
    denominator = len(y) * tmp * (1.0 + 2.0 * a - 2.0 * tmp) / a
    return correct_to_01(numerator / denominator)



def linear(x, m):
    """Shape function."""
    assert shape_args_ok(x, m)
    result = 1.0
    for i in range(1, len(x) - m + 1):
        result *= x[i-1]
    if m != 1:
        result *= 1 - x[len(x) - m]
    return correct_to_01(result)



def convex(x, m):
    """Shape function."""
    assert shape_args_ok(x, m)
    result = 1.0
    for i in range(1, len(x) - m + 1):
        result *= 1.0 - math.cos(x[i - 1] * math.pi / 2.0)
    if m != 1:
        result *= 1.0 - math.sin(x[len(x) - m] * math.pi / 2.0)
    return correct_to_01(result)



def concave(x, m):
    """Shape function."""
    assert shape_args_ok(x, m)
    result = 1.0
    for i in range(1, len(x) - m + 1):
        result *= math.sin(x[i - 1] * math.pi / 2.0)
    if m != 1:
        result *= math.cos(x[len(x) - m] * math.pi / 2.0)
    return correct_to_01(result)



def mixed(x, a, alpha):
    """Shape function."""
    assert vector_in_01(x)
    assert len(x) != 0
    assert a >= 1
    assert alpha > 0.0
    tmp = 2.0 * a * math.pi
    result = math.pow(1.0 - x[0] - math.cos(tmp * x[0] + math.pi / 2.0) / tmp, alpha)
    return correct_to_01(result)



def disc(x, a, alpha, beta):
    """Shape function."""
    assert vector_in_01(x)
    assert len(x) != 0
    assert a >= 1
    assert alpha > 0.0
    assert beta > 0.0
    tmp1 = a * math.pow(x[0], beta) * math.pi
    result = 1.0 - math.pow(x[0], alpha) * math.pow(math.cos(tmp1), 2.0)
    return correct_to_01(result)



class Shape:
    """Abstract base class for shape objects."""

    @staticmethod
    def create_a(m, is_degenerate):
        assert m >= 2
        if is_degenerate:
            a = [0] * (m - 1)
            a[0] = 1
            return a
        else:
            return [1] * (m - 1)


    @staticmethod
    def normalize_z(z, z_max):
        result = []
        for i in range(0, len(z)):
            assert z[i] >= 0.0
            assert z[i] <= z_max[i]
            assert z_max[i] > 0.0
            result.append(z[i] / z_max[i])
        return result


    @staticmethod
    def calculate_x(tp, a):
        assert vector_in_01(tp)
        assert len(tp) != 0
        assert len(a) == len(tp) - 1
        result = []
        for i in range(0, len(tp) - 1):
            assert a[i] == 0 or a[i] == 1
            tmp1 = max(tp[-1], a[i])
            result.append(tmp1 * (tp[i] - 0.5) + 0.5)
        result.append(tp[-1])
        return result


    @staticmethod
    def calculate_f(x, h):
        assert vector_in_01(x)
        assert vector_in_01(h)
        assert len(x) == len(h)
        s = []
        for m in range(1, len(h) + 1):
            s.append(m * 2.0)
        return calculate_f(1.0, x, h, s)


    def __call__(self, tp):
        raise NotImplementedError("Abstract class `Shape` is not callable.")



class WFGBaseProblem(MultiObjectiveTestProblem):
    """The base class for WFG problems."""

    def __init__(self,
                 objective_function,
                 num_objectives,
                 num_variables,
                 k,
                 **kwargs):
        assert num_objectives > 1
        assert k >= 1 and k < num_variables
        assert k % (num_objectives - 1) == 0
        self.max_objectives = None
        self.min_objectives = None
        if num_objectives == 3:
            self.max_objectives = [10.0, 10.0, 10.0]
            self.min_objectives = [0.0, 0.0, 0.0]
        elif num_objectives == 5:
            self.max_objectives = [10.0, 10.0, 10.0, 10.0, 12.0]
            self.min_objectives = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [2.0 * i for i in range(1, num_variables + 1)]
        MultiObjectiveTestProblem.__init__(self,
                                           objective_function,
                                           num_objectives,
                                           **kwargs)
        self.k = k
        self.num_variables = num_variables
        self.is_deterministic = True
        self.do_maximize = False
        if num_objectives <= 4:
            self.default_reference_set_size = 500
        else:
            self.default_reference_set_size = 1000


    @staticmethod
    def args_ok(z, k, m):
        return k >= 1 and k < len(z) and m >= 2 and k % (m - 1) == 0


    def normalize_z(self, z):
        result = []
        for i in range(0, len(z)):
            if z[i] < self.min_bounds[i] or z[i] > self.max_bounds[i]:
                raise BoundConstraintError(z[i], self.min_bounds[i], self.max_bounds[i], "z_"+str(i))
            result.append(z[i] / self.max_bounds[i])
        return result


    @property
    def m(self):
        """The number of objective functions."""
        return self.num_objectives


    def optimal_solution(self, k, l, pos_params):
        """Create one Pareto-optimal solution with given position parameters."""
        # the result vector
        result = []
        result.extend(pos_params)
        # set the distance parameters
        for i in range(k, k + l):
            result.append(0.35)
        # scale to the correct domains
        for i in range(k + l):
            result[i] *= 2.0 * (i + 1)
        ind = Individual()
        ind.phenome = result
        return ind


    def rand_optimal_solution(self, k, l, rand_generator=None):
        """Generate random position parameters and build optimal solution."""
        params = []
        if rand_generator is None:
            rand_generator = random
        for _ in range(k):
            params.append(rand_generator.random())
        return self.optimal_solution(k, l, params)


    def get_optimal_solutions(self, max_number):
        solutions = []
        rand_generator = random.Random()
        rand_generator.seed(2)
        optimal_solution = self.optimal_solution
        rand_optimal_solution = self.rand_optimal_solution
        k = self.k
        l = self.num_variables - k
        # generate full factorial sample of position parameters
        param_combis = [[]]
        for _ in range(k):
            new_combis = []
            for combi in param_combis:
                combi_copy = copy.deepcopy(combi)
                combi.append(1.0)
                combi_copy.append(0.0)
                new_combis.append(combi_copy)
            param_combis.extend(new_combis)
        # build solutions for these parameters
        for pos_params in param_combis:
            solutions.append(optimal_solution(k, l, pos_params))
        # fill up with random optimal solutions
        while len(solutions) < max_number:
            solutions.append(rand_optimal_solution(k, l, rand_generator))
        # assert length is not exceeded
        return solutions[0:max_number]



class WFG1(WFGBaseProblem):
    """The WFG1 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        self.shape = self.WFG1Shape()


    def optimal_solution(self, k, l, pos_params):
        """Create one Pareto-optimal solution with given position parameters."""
        # the result vector
        result = []
        result.extend(pos_params)
        # set the distance parameters
        for i in range(k, k + l):
            result.append(0.35)
        # scale to the correct domains
        for i in range(k + l):
            result[i] *= 2.0 * (i + 1)
        ind = Individual()
        ind.phenome = result
        return ind


    def rand_optimal_solution(self, k, l, rand_generator=None):
        """Generate random position parameters and build optimal solution."""
        params = []
        if rand_generator is None:
            rand_generator = random
        for _ in range(k):
            # account for polynomial bias
            params.append(pow(rand_generator.random(), 50.0))
        return self.optimal_solution(k, l, params)


    @staticmethod
    def transition1(y, k):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        t = y[0:k]
        for i in range(k, n):
            t.append(s_linear(y[i], 0.35))
        return t


    @staticmethod
    def transition2(y, k):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        t = y[0:k]
        for i in range(k, n):
            t.append(b_flat(y[i], 0.8, 0.75, 0.85))
        return t


    @staticmethod
    def transition3(y):
        n = len(y)
        assert vector_in_01(y)
        t = []
        for i in range(0, n):
            t.append(b_poly(y[i], 0.02))
        return t


    @staticmethod
    def transition4(y, k, m):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        assert m >= 2
        assert k % (m - 1) == 0
        w = []
        for i in range(1, n + 1):
            w.append(2.0 * i)
        t = []
        for i in range(1, m):
            head = int((i - 1) * k / (m - 1))
            tail = int(i * k / (m - 1))
            y_sub = y[head:tail]
            w_sub = w[head:tail]
            t.append(r_sum(y_sub, w_sub))
        y_sub = y[k:n]
        w_sub = w[k:n]
        t.append(r_sum(y_sub, w_sub))
        return t


    class WFG1Shape(Shape):

        def __call__(self, tp):
            assert vector_in_01(tp)
            assert len(tp) >= 2
            a = self.create_a(len(tp), False)
            x = self.calculate_x(tp, a)
            h = []
            for m in range(1, len(tp)):
                h.append(convex(x, m))
            h.append(mixed(x, 5, 1.0))
            return self.calculate_f(x, h)


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        y = self.normalize_z(phenome)
        y = self.transition1(y, self.k)
        y = self.transition2(y, self.k)
        y = self.transition3(y)
        y = self.transition4(y, self.k, self.m)
        return self.shape(y)



class WFG2(WFGBaseProblem):
    """The WFG2 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        assert (num_variables - k) % 2 == 0
        self.shape = self.WFG2Shape()
        self.transition1 = WFG1.transition1


    @staticmethod
    def transition2(y, k):
        n = len(y)
        l = n - k
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        assert l % 2 == 0
        t = []
        for i in range(k):
            t.append(y[i])
        for i in range(k + 1, int(k + l / 2 + 1)):
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y_sub = y[head:tail]
            t.append(r_nonsep(y_sub, 2))
        return t


    @staticmethod
    def transition3(y, k, m):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        assert m >= 2
        assert k % (m - 1) == 0
        w = [1.0] * n
        t = []
        for i in range(1, m):
            head = int((i - 1) * k / (m - 1))
            tail = int(i * k / (m - 1))
            y_sub = y[head:tail]
            w_sub = w[head:tail]
            t.append(r_sum(y_sub, w_sub))
        y_sub = y[k:n]
        w_sub = w[k:n]
        t.append(r_sum(y_sub, w_sub))
        return t


    class WFG2Shape(Shape):

        def __call__(self, tp):
            assert vector_in_01(tp)
            assert len(tp) >= 2
            a = self.create_a(len(tp), False)
            x = self.calculate_x(tp, a)
            h = []
            for m in range(1, len(tp)):
                h.append(convex(x, m))
            h.append(disc(x, 5, 1.0, 1.0))
            return self.calculate_f(x, h)


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        assert (len(phenome) - self.k) % 2 == 0
        y = self.normalize_z(phenome)
        y = self.transition1(y, self.k)
        y = self.transition2(y, self.k)
        y = self.transition3(y, self.k, self.m)
        return self.shape(y)



class WFG3(WFGBaseProblem):
    """The WFG3 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        assert (num_variables - k) % 2 == 0
        self.shape = self.WFG3Shape()
        self.transition1 = WFG1.transition1
        self.transition2 = WFG2.transition2
        self.transition3 = WFG2.transition3


    class WFG3Shape(Shape):

        def __call__(self, tp):
            assert vector_in_01(tp)
            assert len(tp) >= 2
            a = self.create_a(len(tp), True)
            x = self.calculate_x(tp, a)
            h = []
            for m in range(1, len(tp) + 1):
                h.append(linear(x, m))
            return self.calculate_f(x, h)


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        assert (len(phenome) - self.k) % 2 == 0
        y = self.normalize_z(phenome)
        y = self.transition1(y, self.k)
        y = self.transition2(y, self.k)
        y = self.transition3(y, self.k, self.m)
        return self.shape(y)



class WFG4(WFGBaseProblem):
    """The WFG4 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        assert k % (num_objectives - 1) == 0
        self.shape = self.WFG4Shape()
        self.transition2 = WFG2.transition3


    @staticmethod
    def transition1(y):
        n = len(y)
        assert vector_in_01(y)
        t = []
        for i in range(0, n):
            t.append(s_multi(y[i], 30, 10, 0.35))
        return t


    class WFG4Shape(Shape):

        def __call__(self, tp):
            assert vector_in_01(tp)
            assert len(tp) >= 2
            a = self.create_a(len(tp), False)
            x = self.calculate_x(tp, a)
            h = []
            for m in range(1, len(tp) + 1):
                h.append(concave(x, m))
            return self.calculate_f(x, h)


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        y = self.normalize_z(phenome)
        y = self.transition1(y)
        y = self.transition2(y, self.k, self.m)
        return self.shape(y)



class WFG5(WFGBaseProblem):
    """The WFG5 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        self.shape = WFG4.WFG4Shape()
        self.transition2 = WFG2.transition3


    @staticmethod
    def transition1(y):
        n = len(y)
        assert vector_in_01(y)
        t = []
        for i in range(0, n):
            t.append(s_decept(y[i], 0.35, 0.001, 0.05))
        return t


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        y = self.normalize_z(phenome)
        y = self.transition1(y)
        y = self.transition2(y, self.k, self.m)
        return self.shape(y)



class WFG6(WFGBaseProblem):
    """The WFG6 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        self.shape = WFG4.WFG4Shape()
        self.transition1 = WFG1.transition1


    @staticmethod
    def transition2(y, k, m):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        assert m >= 2
        assert k % (m - 1) == 0
        t = []
        for i in range(1, m):
            head = int((i - 1) * k / (m - 1))
            tail = int(i * k / (m - 1))
            y_sub = y[head:tail]
            t.append(r_nonsep(y_sub, int(k / (m - 1))))
        y_sub = y[k:n]
        t.append(r_nonsep(y_sub, n - k))
        return t


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        y = self.normalize_z(phenome)
        y = self.transition1(y, self.k)
        y = self.transition2(y, self.k, self.m)
        return self.shape(y)



class WFG7(WFGBaseProblem):
    """The WFG7 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        self.shape = WFG4.WFG4Shape()
        self.transition2 = WFG1.transition1
        self.transition3 = WFG2.transition3


    @staticmethod
    def transition1(y, k):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        w = [1.0] * n
        t = []
        for i in range(k):
            y_sub = y[i+1:n]
            w_sub = w[i+1:n]
            u = r_sum(y_sub, w_sub)
            t.append(b_param(y[i], u, 0.98/49.98, 0.02, 50))
        for i in range(k, n):
            t.append(y[i])
        return t


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        y = self.normalize_z(phenome)
        y = self.transition1(y, self.k)
        y = self.transition2(y, self.k)
        y = self.transition3(y, self.k, self.m)
        return self.shape(y)



class WFG8(WFGBaseProblem):
    """The WFG8 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        self.shape = WFG4.WFG4Shape()
        self.transition2 = WFG1.transition1
        self.transition3 = WFG2.transition3
        if num_objectives <= 4:
            self.default_reference_set_size = 500
        else:
            self.default_reference_set_size = 1000


    def optimal_solution(self, k, l, pos_params):
        """Create one Pareto-optimal solution with given position parameters."""
        result = []
        result.extend(pos_params)
        # calculate the distance parameters
        for i in range(k, k + l):
            w = [1.0] * len(result)
            u = r_sum(result, w)
            tmp1 = abs(math.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0 * u) * tmp1)
            result.append(pow(0.35, pow(tmp2, -1.0)))
        # scale to the correct domains
        for i in range(k + l):
            result[i] *= 2.0 * (i + 1)
        ind = Individual()
        ind.phenome = result
        return ind


    @staticmethod
    def transition1(y, k):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        w = [1.0] * n
        t = y[0:k]
        for i in range(k, n):
            y_sub = y[0:i]
            w_sub = w[0:i]
            u = r_sum(y_sub, w_sub)
            t.append(b_param(y[i], u, 0.98 / 49.98, 0.02, 50))
        return t


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        y = self.normalize_z(phenome)
        y = self.transition1(y, self.k)
        y = self.transition2(y, self.k)
        y = self.transition3(y, self.k, self.m)
        return self.shape(y)



class WFG9(WFGBaseProblem):
    """The WFG9 problem."""

    def __init__(self, num_objectives, num_variables, k, **kwargs):
        WFGBaseProblem.__init__(self,
                                self.objective_function,
                                num_objectives,
                                num_variables,
                                k,
                                **kwargs)
        self.shape = WFG4.WFG4Shape()
        self.transition3 = WFG6.transition2
        if num_objectives <= 4:
            self.default_reference_set_size = 500
        else:
            self.default_reference_set_size = 1000


    def optimal_solution(self, k, l, pos_params):
        """Create one Pareto-optimal solution with given position parameters."""
        result = []
        result.extend(pos_params)
        result.extend([0.0] * l)
        # calculate the distance parameters
        result[k + l - 1] = 0.35
        for i in range(k + l - 2, k - 1, -1):
            result_sub = []
            for j in range(i + 1, k + l):
                result_sub.append(result[j])
            w = [1.0] * len(result_sub)
            tmp1 = r_sum(result_sub, w)
            result[i] = pow(0.35, pow(0.02 + 1.96 * tmp1, -1.0))
        # scale to the correct domains
        for i in range(k + l):
            result[i] *= 2.0 * (i + 1)
        ind = Individual()
        ind.phenome = result
        return ind


    @staticmethod
    def transition1(y):
        n = len(y)
        assert vector_in_01(y)
        w = [1.0] * n
        t = []
        for i in range(0, n - 1):
            y_sub = y[i+1:n]
            w_sub = w[i+1:n]
            u = r_sum(y_sub, w_sub)
            t.append(b_param(y[i], u, 0.98 / 49.98, 0.02, 50))
        t.append(y[-1])
        return t


    @staticmethod
    def transition2(y, k):
        n = len(y)
        assert vector_in_01(y)
        assert k >= 1
        assert k < n
        t = []
        for i in range(0, k):
            t.append(s_decept(y[i], 0.35, 0.001, 0.05))
        for i in range(k, n):
            t.append(s_multi(y[i], 30, 95, 0.35))
        return t


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        assert self.args_ok(phenome, self.k, self.m)
        y = self.normalize_z(phenome)
        y = self.transition1(y)
        y = self.transition2(y, self.k)
        y = self.transition3(y, self.k, self.m)
        return self.shape(y)



class WFG(list):
    """The test problem collection WFG as defined in [Huband2006]_.

    This class inherits from :class:`list` and fills itself with the nine
    WFG problems with the chosen configuration. The arguments to the
    constructor are passed through to the problem classes.

    References
    ----------
    .. [Huband2006] Huband, S.; Hingston, P.; Barone, L.; While, L. (2006).
        A review of multiobjective test problems and a scalable test
        problem toolkit. IEEE Transactions on Evolutionary Computation,
        vol.10, no.5, pp. 477-506.

    """
    def __init__(self, num_objectives, num_variables, k, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_objectives : int
            The number of objectives for the optimization problems.
        num_variables : int
            The number of decision variables of the problems.
        k : int
            The number of position related parameters, which are a part
            of the phenome. Thus it must hold ``k < num_variables``.
            Furthermore, k must be a multiple of ``num_objectives - 1``.
            Huband et al. recommend ``k = 4`` for two objectives and
            ``k = 2 * (m - 1)`` for m objectives.
        kwargs
            Arbitrary keyword arguments, passed through to the constructors
            of the single WFG problems.

        """
        wfg1 = WFG1(num_objectives, num_variables, k, **kwargs)
        wfg2 = WFG2(num_objectives, num_variables, k, **kwargs)
        wfg3 = WFG3(num_objectives, num_variables, k, **kwargs)
        wfg4 = WFG4(num_objectives, num_variables, k, **kwargs)
        wfg5 = WFG5(num_objectives, num_variables, k, **kwargs)
        wfg6 = WFG6(num_objectives, num_variables, k, **kwargs)
        wfg7 = WFG7(num_objectives, num_variables, k, **kwargs)
        wfg8 = WFG8(num_objectives, num_variables, k, **kwargs)
        wfg9 = WFG9(num_objectives, num_variables, k, **kwargs)
        list.__init__(self, [wfg1, wfg2, wfg3, wfg4, wfg5, wfg6, wfg7, wfg8, wfg9])
