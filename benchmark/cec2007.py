"""
This set of test problems was compiled for the Special Session & Competition
on Performance Assessment of Multi-Objective Optimization Algorithms at the
Congress on Evolutionary Computation (CEC), Singapore, 25-28 September 2007.
The mathematical definitions are given in [Huang2007]_. Additionally, a C
implementation was provided for the participants. When the problems were
later reimplemented in Python, several bugs surfaced in the C implementation.
The bugs were originally found in 2009 and documented in appendix B.2.4 of
[Wessing2009]_, but the code was not published back then. In this
implementation, it is possible to switch between the original/buggy and the
fixed behavior.

References
----------
.. [Huang2007] V. L. Huang, A. K. Qin, K. Deb, E. Zitzler,
    P. N. Suganthan, J. J. Liang, M. Preuss and S. Huband (2007).
    Problem Definitions for Performance Assessment of Multi-objective
    Optimization Algorithms. Special Session on Constrained
    Real-Parameter Optimization, Technical Report, Nanyang
    Technological University, Singapore, 2007.
    http://web.mysites.ntu.edu.sg/epnsugan/PublicSite/Shared%20Documents/CEC-2007/CEC-07-TR-13-Feb.pdf

.. [Wessing2009] Simon Wessing. Towards Optimal Parameterizations of the
    S-Metric Selection Evolutionary Multi-Objective Algorithm. Diploma
    thesis, Algorithm Engineering Report TR09-2-006, Technische Universitaet
    Dortmund, 2009.
    https://ls11-www.cs.uni-dortmund.de/_media/techreports/tr09-06.pdf

"""
import math
import copy
import random

import numpy as np

from benchmark.base import Individual, BoundConstraintsChecker
from benchmark.multiobjective import MultiObjectiveTestProblem
from benchmark.zdt import ZDT1, ZDT2, ZDT4, ZDT6
from benchmark.dtlz import DTLZ2, DTLZ3
from benchmark.wfg import WFG1, WFG8, WFG9

__all__ = ["CEC2007", "OKA2", "SYMPART", "S_ZDT1", "S_ZDT2", "S_ZDT4", "R_ZDT4",
           "S_ZDT6", "S_DTLZ2", "R_DTLZ2", "S_DTLZ3", "WFG1", "WFG8", "WFG9"]

class OKA2(MultiObjectiveTestProblem):
    """The OKA2 problem.

    This problem was defined in [Okabe2004]_.

    References
    ----------
    .. [Okabe2004] Tatsuya Okabe, Yaochu Jin, Markus Olhofer, Bernhard
        Sendhoff (2004). On Test Functions for Evolutionary Multi-objective
        Optimization. Parallel Problem Solving from Nature - PPSN VIII,
        LNCS Vol. 3242, pp 792-802, Springer.
        https://dx.doi.org/10.1007/978-3-540-30217-9_80

    """
    def __init__(self, is_orig_behavior=False, phenome_preprocessor=None, **kwargs):
        """Constructor.

        Note that the C implementation of CEC 2007 contained a bug, using
        abs where fabs should be used.

        Parameters
        ----------
        is_orig_behavior : bool, optional
            A flag indicating if the behavior of the C implementation
            should be used.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the base class.

        """
        self.default_reference_set_size = 300
        self.is_orig_behavior = is_orig_behavior
        self.min_bounds = [-math.pi, -5.0, -5.0]
        self.max_bounds = [math.pi, 5.0, 5.0]
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        self.num_variables = 3
        self.do_maximize = False
        self.is_deterministic = True
        MultiObjectiveTestProblem.__init__(self,
                                           self.objective_function,
                                           num_objectives=2,
                                           phenome_preprocessor=preprocessor,
                                           **kwargs)


    def get_optimal_solutions(self, max_number):
        solutions = []
        for i in range(max_number):
            xi = -math.pi + (2.0 * math.pi * i / (max_number - 1))
            xi = max(xi, -math.pi)
            xi = min(xi, math.pi)
            ind = Individual([xi, 5.0 * math.cos(xi), 5.0 * math.sin(xi)])
            solutions.append(ind)
        return solutions


    def objective_function(self, phenome):
        x = phenome
        x0_part = ((x[0] + math.pi) ** 2) / (4.0 * math.pi * math.pi)
        x1_part = pow(abs(x[1] - 5.0 * math.cos(x[0])), 1.0 / 3.0)
        x2_part = pow(abs(x[2] - 5.0 * math.sin(x[0])), 1.0 / 3.0)
        if self.is_orig_behavior:
            # the C implementation of CEC 2007 contains a bug,
            # using abs where fabs should be used
            # manually convert to int here to simulate the behavior
            x1_part = pow(int(abs(x[1] - 5.0 * math.cos(x[0]))), 1.0 / 3.0)
            x2_part = pow(int(abs(x[2] - 5.0 * math.sin(x[0]))), 1.0 / 3.0)
        return [phenome[0], 1 - x0_part + x1_part + x2_part]



class SYMPART(MultiObjectiveTestProblem):
    """The CEC 2007 SYMPART variant.

    This SYMPART variant was modified to work on 30 decision variables.
    The original problem in [Rudolph2007]_ only had a two dimensional
    search space. It is unclear how this modified problem behaves compared
    to the original one with two decision variables.

    References
    ----------
    .. [Rudolph2007] Guenter Rudolph, Boris Naujoks, Mike Preuss (2007).
        Capabilities of EMOA to Detect and Preserve Equivalent Pareto
        Subsets. In: Evolutionary Multi-Criterion Optimization, LNCS Vol.
        4403, pp 36-50, Springer. https://dx.doi.org/10.17877/DE290R-590

    """
    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        self.default_reference_set_size = 300
        self.rotation_angle = math.pi / 4.0
        self.a = 1.0
        self.b = 10.0
        self.c = 8.0
        self.c2 = self.c + 2.0 * self.a
        self.num_variables = num_variables
        self.min_bounds = [-20.0] * num_variables
        self.max_bounds = [20.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        self.do_maximize = False
        self.is_deterministic = True
        MultiObjectiveTestProblem.__init__(self,
                                           [self.f1, self.f2],
                                           num_objectives=2,
                                           phenome_preprocessor=preprocessor,
                                           **kwargs)


    @staticmethod
    def rotate(phenome, angle_radians):
        """Rotate the candidate solution."""
        ret = copy.deepcopy(phenome)
        sin = math.sin(angle_radians)
        cos = math.cos(angle_radians)
        for dim in range(0, len(phenome) - 1, 2):
            h1 = ret[dim]
            ret[dim] = cos * h1 - sin * ret[dim+1]
            ret[dim+1] = sin * h1 + cos * ret[dim+1]
        return ret


    @staticmethod
    def signum(number):
        if number < 0:
            return -1
        elif number > 0:
            return 1
        else:
            return 0


    def t1(self, value):
        return self.signum(value) * math.ceil((abs(value) - self.c2 / 2.0) / self.c2)


    def t2(self, value):
        return self.signum(value) * math.ceil((abs(value) - self.b / 2.0) / self.b)


    def get_optimal_solutions(self, max_number):
        """Generate optimal solutions.

        This method does not exploit any of the problem's symmetry properties.
        The Pareto-set was only discovered empirically.

        """
        rand_generator = random.Random()
        rand_generator.seed(1)
        solutions = []
        sqrt_two = math.sqrt(2)
        for i in range(max_number):
            value1 = -(6.3639610306789271 + (sqrt_two * i / (max_number - 1)))
            value2 = -value1
            ind = Individual()
            ind.phenome = [value1, value2] * int(self.num_variables / 2)
            solutions.append(ind)
        return solutions


    def f1(self, phenome):
        x_prime = self.rotate(phenome, self.rotation_angle)
        # signum restricts positions t_i to {-1, 0, 1} => 9 tiles
        t1 = self.signum(self.t1(x_prime[0]))
        t2 = self.signum(self.t2(x_prime[1]))
        ret = 0.0
        for i in range(self.num_variables):
            if i % 2 == 0:
                ret += (x_prime[i] + self.a - t1 * self.c2) ** 2
            else:
                ret += (x_prime[i] - t2 * self.b) ** 2
        return ret / self.num_variables


    def f2(self, phenome):
        x_prime = self.rotate(phenome, self.rotation_angle)
        # signum restricts positions t_i to {-1, 0, 1} => 9 tiles
        t1 = self.signum(self.t1(x_prime[0]))
        t2 = self.signum(self.t2(x_prime[1]))
        ret = 0.0
        for i in range(len(x_prime)):
            if i % 2 == 0:
                ret += (x_prime[i] - self.a - t1 * self.c2) ** 2
            else:
                ret += (x_prime[i] - t2 * self.b) ** 2
        return ret / self.num_variables



class ExtendedProblem(MultiObjectiveTestProblem):
    """Base class for all extended ZDT and DTLZ problems."""

    @staticmethod
    def p_sum(penalties, used_indices):
        """Calculate the penalty value for the used variables."""
        used_numbers = []
        for i in used_indices:
            used_numbers.append(penalties[i])
        return math.sqrt(sum(x ** 2 for x in used_numbers))


    @staticmethod
    def stretch(t):
        """Manipulate an objective value if boundaries are violated.

        Parameters
        ----------
        t : float
            The output of the function
            :func:`p_sum <benchmark.cec2007.ExtendedProblem.p_sum>`.

        """
        return 2.0 / (1.0 + math.exp(-t))


    def objective_function(self, phenome):
        z = self.calc_z(phenome)
        z_prime = self.calc_z_prime(z)
        penalties = self.calc_penalties(z)
        objective_values = self.problem(z_prime)
        for i, obj_value in enumerate(objective_values):
            used_indices = self.used_indices_per_dim[i]
            objective_values[i] = (obj_value + 1.0) * self.stretch(self.p_sum(penalties, used_indices))
        return objective_values



class ShiftedProblem(ExtendedProblem):
    """Base class for all shifted ZDT and DTLZ problems.

    Inherits from :class:`ExtendedProblem<benchmark.cec2007.ExtendedProblem>`.

    """

    def cut(self, num_variables):
        """Truncate the problem's various vectors to the needed dimension."""
        self.num_variables = num_variables
        # offsets
        self.o = self.o[0:num_variables]
        self.d = self.d[0:num_variables]
        # scaling factors
        self.lambdas = self.lambdas[0:num_variables]
        # boundaries
        self.min_bounds = self.min_bounds[0:num_variables]
        self.max_bounds = self.max_bounds[0:num_variables]


    def calc_z(self, phenome):
        """Calculate the shifted values from the candidate solution."""
        z = []
        for i in range(len(phenome)):
            z.append(phenome[i] - self.o[i])
        return z


    def calc_penalties(self, z):
        """Calculate the penalties from the z values."""
        p = []
        for i in range(len(z)):
            if z[i] >= 0:
                p.append(0.0)
            else:
                p.append(abs(z[i]) / self.d[i])
        return p


    def calc_z_prime(self, z):
        """Calculate the z' values from the z values."""
        z_prime = []
        for i in range(len(z)):
            if z[i] >= 0:
                z_prime.append(z[i])
            else:
                z_prime.append(-self.lambdas[i] * z[i])
        return z_prime



class RotatedProblem(ExtendedProblem):
    """Base class for all rotated ZDT and DTLZ problems.

    Inherits from :class:`ExtendedProblem<benchmark.cec2007.ExtendedProblem>`.

    """
    def calc_z(self, phenome):
        """Calculate the rotated values from the candidate solution."""
        # multiply matrix with column vector (= rotate)
        z = np.dot(self.rotation_matrix, phenome).tolist()
        return z



class S_ZDT1(ShiftedProblem):
    """A shifted ZDT1 problem."""

    def __init__(self, num_variables, is_orig_behavior=False, **kwargs):
        # the problem that is extended here
        self.problem = ZDT1(num_variables)
        self.is_orig_behavior = is_orig_behavior
        self.used_indices_per_dim = [[0]]
        if is_orig_behavior:
            self.used_indices_per_dim.append(list(range(1, num_variables)))
        else:
            self.used_indices_per_dim.append(list(range(num_variables)))
        self.max_objectives = [2.0, 10.0]
        self.min_objectives = [1.0, 1.0]
        self.do_maximize = False
        self.is_deterministic = True
        # call superclass constructor
        ShiftedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=2,
                                **kwargs)
        self.default_reference_set_size = 300
        self.o = [0.950, 0.231, 0.607, 0.486, 0.891, 0.762, 0.457, 0.019, 0.821,
                  0.445, 0.615, 0.792, 0.922, 0.738, 0.176, 0.406, 0.936, 0.917,
                  0.410, 0.894, 0.058, 0.353, 0.813, 0.010, 0.139, 0.203, 0.199,
                  0.604, 0.272, 0.199, 0.015, 0.747, 0.445, 0.932, 0.466, 0.419,
                  0.846, 0.525, 0.203, 0.672, 0.838, 0.020, 0.681, 0.380, 0.832,
                  0.503, 0.710, 0.429, 0.305, 0.190, 0.193, 0.682, 0.303, 0.542,
                  0.151, 0.698, 0.378, 0.860, 0.854, 0.594, 0.497, 0.900, 0.822,
                  0.645, 0.818, 0.660, 0.342, 0.290, 0.341, 0.534, 0.727, 0.309,
                  0.839, 0.568, 0.370, 0.703, 0.547, 0.445, 0.695, 0.621, 0.795,
                  0.957, 0.523, 0.880, 0.173, 0.980, 0.271, 0.252, 0.876, 0.737,
                  0.137, 0.012, 0.894, 0.199, 0.299, 0.661, 0.284, 0.469, 0.065,
                  0.988]
        self.d = [0.155, 0.119, 0.185, 0.064, 0.070, 0.203, 0.166, 0.151, 0.219,
                  0.083, 0.161, 0.057, 0.169, 0.072, 0.135, 0.114, 0.241, 0.129,
                  0.230, 0.181, 0.195, 0.058, 0.152, 0.107, 0.119, 0.219, 0.203,
                  0.212, 0.126, 0.238, 0.164, 0.236, 0.086, 0.087, 0.188, 0.244,
                  0.146, 0.124, 0.218, 0.196, 0.120, 0.140, 0.120, 0.207, 0.205,
                  0.114, 0.107, 0.228, 0.097, 0.208, 0.105, 0.120, 0.135, 0.196,
                  0.183, 0.111, 0.081, 0.236, 0.183, 0.171, 0.063, 0.160, 0.246,
                  0.091, 0.064, 0.188, 0.222, 0.082, 0.174, 0.114, 0.155, 0.209,
                  0.179, 0.130, 0.184, 0.200, 0.108, 0.143, 0.116, 0.155, 0.078,
                  0.189, 0.128, 0.171, 0.168, 0.226, 0.061, 0.250, 0.201, 0.226,
                  0.065, 0.194, 0.114, 0.159, 0.135, 0.239, 0.167, 0.204, 0.095,
                  0.091]
        self.lambdas = [3.236, 4.201, 2.701, 7.775, 7.148, 2.465, 3.020, 3.322,
                        2.285, 6.004, 3.106, 8.801, 2.956, 6.918, 3.708, 4.403,
                        2.077, 3.884, 2.171, 2.760, 2.567, 8.636, 3.299, 4.666,
                        4.208, 2.285, 2.469, 2.362, 3.963, 2.099, 3.052, 2.120,
                        5.809, 5.753, 2.654, 2.053, 3.427, 4.046, 2.297, 2.548,
                        4.151, 3.577, 4.165, 2.417, 2.445, 4.405, 4.668, 2.192,
                        5.147, 2.398, 4.743, 4.157, 3.710, 2.553, 2.733, 4.519,
                        6.159, 2.121, 2.738, 2.926, 7.918, 3.130, 2.034, 5.495,
                        7.846, 2.666, 2.253, 6.068, 2.881, 4.388, 3.220, 2.389,
                        2.791, 3.832, 2.711, 2.496, 4.624, 3.505, 4.297, 3.223,
                        6.428, 2.644, 3.920, 2.917, 2.978, 2.215, 8.138, 2.000,
                        2.490, 2.215, 7.711, 2.574, 4.395, 3.135, 3.705, 2.091,
                        3.003, 2.456, 5.241, 5.499]
        self.min_bounds = [0.795, 0.112, 0.422, 0.422, 0.821, 0.559, 0.291,
                           -0.132, 0.602, 0.362, 0.454, 0.735, 0.753, 0.666,
                           0.041, 0.292, 0.695, 0.788, 0.18, 0.713, -0.137,
                           0.295, 0.661, -0.097, 0.02, -0.016, -0.004, 0.392,
                           0.146, -0.039, -0.149, 0.511, 0.359, 0.845, 0.278,
                           0.175, 0.7, 0.401, -0.015, 0.476, 0.718, -0.12,
                           0.561, 0.173, 0.627, 0.389, 0.603, 0.201, 0.208,
                           -0.018, 0.088, 0.562, 0.168, 0.346, -0.032, 0.587,
                           0.297, 0.624, 0.671, 0.423, 0.434, 0.74, 0.576,
                           0.554, 0.754, 0.472, 0.12, 0.208, 0.167, 0.42, 0.572,
                           0.1, 0.66, 0.438, 0.186, 0.503, 0.439, 0.302, 0.579,
                           0.466, 0.717, 0.768, 0.395, 0.709, 0.005, 0.754,
                           0.21, 0.002, 0.675, 0.511, 0.072, -0.182, 0.78, 0.04,
                           0.164, 0.422, 0.117, 0.265, -0.03, 0.897]
        self.max_bounds = [1.950, 1.231, 1.607, 1.486, 1.891, 1.762, 1.457,
                           1.019, 1.821, 1.445, 1.615, 1.792, 1.922, 1.738,
                           1.176, 1.406, 1.936, 1.917, 1.41, 1.894, 1.058,
                           1.353, 1.813, 1.01, 1.139, 1.203, 1.199, 1.604,
                           1.272, 1.199, 1.015, 1.747, 1.445, 1.932, 1.466,
                           1.419, 1.846, 1.525, 1.203, 1.672, 1.838, 1.02,
                           1.681, 1.38, 1.832, 1.503, 1.71, 1.429, 1.305, 1.19,
                           1.193, 1.682, 1.303, 1.542, 1.151, 1.698, 1.378,
                           1.86, 1.854, 1.594, 1.497, 1.9, 1.822, 1.645, 1.818,
                           1.66, 1.342, 1.29, 1.341, 1.534, 1.727, 1.309, 1.839,
                           1.568, 1.37, 1.703, 1.547, 1.445, 1.695, 1.621,
                           1.795, 1.957, 1.523, 1.88, 1.173, 1.98, 1.271, 1.252,
                           1.876, 1.737, 1.137, 1.012, 1.894, 1.199, 1.299,
                           1.661, 1.284, 1.469, 1.065, 1.988]
        # truncate the above vectors to the number of used dimensions
        self.cut(num_variables)


    def get_optimal_solutions(self, max_number):
        solutions = []
        for i in range(max_number):
            phenome = copy.deepcopy(self.o)
            phenome[0] = self.o[0] + float(i) / float(max_number - 1)
            phenome[0] = max(phenome[0], self.min_bounds[0])
            phenome[0] = min(phenome[0], self.max_bounds[0])
            solutions.append(Individual(phenome))
        return solutions



class S_ZDT2(ShiftedProblem):
    """A shifted ZDT2 problem."""

    def __init__(self, num_variables, is_orig_behavior=False, **kwargs):
        # the problem that is extended here
        self.problem = ZDT2(num_variables)
        self.is_orig_behavior = is_orig_behavior
        self.used_indices_per_dim = [[0]]
        if is_orig_behavior:
            self.used_indices_per_dim.append(list(range(1, num_variables)))
        else:
            self.used_indices_per_dim.append(list(range(num_variables)))
        self.max_objectives = [2.0, 10.0]
        self.min_objectives = [1.0, 1.0]
        self.do_maximize = False
        self.is_deterministic = True
        # call superclass constructor
        ShiftedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=2,
                                **kwargs)
        self.default_reference_set_size = 300
        self.o = [0.583, 0.424, 0.516, 0.334, 0.433, 0.226, 0.580, 0.760, 0.530,
                  0.641, 0.209, 0.380, 0.783, 0.681, 0.461, 0.568, 0.794, 0.059,
                  0.603, 0.050, 0.415, 0.305, 0.874, 0.015, 0.768, 0.971, 0.990,
                  0.789, 0.439, 0.498, 0.214, 0.643, 0.320, 0.960, 0.727, 0.412,
                  0.745, 0.268, 0.440, 0.933, 0.683, 0.213, 0.839, 0.629, 0.134,
                  0.207, 0.607, 0.630, 0.370, 0.575, 0.451, 0.044, 0.027, 0.313,
                  0.013, 0.384, 0.683, 0.093, 0.035, 0.612, 0.609, 0.016, 0.016,
                  0.190, 0.587, 0.058, 0.368, 0.631, 0.718, 0.693, 0.084, 0.454,
                  0.442, 0.353, 0.154, 0.676, 0.699, 0.728, 0.478, 0.555, 0.121,
                  0.451, 0.716, 0.893, 0.273, 0.255, 0.866, 0.232, 0.805, 0.908,
                  0.232, 0.239, 0.050, 0.078, 0.641, 0.191, 0.844, 0.174, 0.171,
                  0.994]
        self.d = [0.155, 0.119, 0.185, 0.064, 0.070, 0.203, 0.166, 0.151, 0.219,
                  0.083, 0.161, 0.057, 0.169, 0.072, 0.135, 0.114, 0.241, 0.129,
                  0.230, 0.181, 0.195, 0.058, 0.152, 0.107, 0.119, 0.219, 0.203,
                  0.212, 0.126, 0.238, 0.164, 0.236, 0.086, 0.087, 0.188, 0.244,
                  0.146, 0.124, 0.218, 0.196, 0.120, 0.140, 0.120, 0.207, 0.205,
                  0.114, 0.107, 0.228, 0.097, 0.208, 0.105, 0.120, 0.135, 0.196,
                  0.183, 0.111, 0.081, 0.236, 0.183, 0.171, 0.063, 0.160, 0.246,
                  0.091, 0.064, 0.188, 0.222, 0.082, 0.174, 0.114, 0.155, 0.209,
                  0.179, 0.130, 0.184, 0.200, 0.108, 0.143, 0.116, 0.155, 0.078,
                  0.189, 0.128, 0.171, 0.168, 0.226, 0.061, 0.250, 0.201, 0.226,
                  0.065, 0.194, 0.114, 0.159, 0.135, 0.239, 0.167, 0.204, 0.095,
                  0.091]
        self.lambdas = [3.236, 4.201, 2.701, 7.775, 7.148, 2.465, 3.020, 3.322,
                        2.285, 6.004, 3.106, 8.801, 2.956, 6.918, 3.708, 4.403,
                        2.077, 3.884, 2.171, 2.760, 2.567, 8.636, 3.299, 4.666,
                        4.208, 2.285, 2.469, 2.362, 3.963, 2.099, 3.052, 2.120,
                        5.809, 5.753, 2.654, 2.053, 3.427, 4.046, 2.297, 2.548,
                        4.151, 3.577, 4.165, 2.417, 2.445, 4.405, 4.668, 2.192,
                        5.147, 2.398, 4.743, 4.157, 3.710, 2.553, 2.733, 4.519,
                        6.159, 2.121, 2.738, 2.926, 7.918, 3.130, 2.034, 5.495,
                        7.846, 2.666, 2.253, 6.068, 2.881, 4.388, 3.220, 2.389,
                        2.791, 3.832, 2.711, 2.496, 4.624, 3.505, 4.297, 3.223,
                        6.428, 2.644, 3.920, 2.917, 2.978, 2.215, 8.138, 2.000,
                        2.490, 2.215, 7.711, 2.574, 4.395, 3.135, 3.705, 2.091,
                        3.003, 2.456, 5.241, 5.499]
        self.min_bounds = [0.428, 0.305, 0.331, 0.270, 0.363, 0.023, 0.414,
                           0.609, 0.311, 0.558, 0.048, 0.323, 0.614, 0.609,
                           0.326, 0.454, 0.553, -0.07, 0.373, -0.131, 0.22,
                           0.247, 0.722, -0.092, 0.649, 0.752, 0.787, 0.577,
                           0.313, 0.26, 0.05, 0.407, 0.234, 0.873, 0.539, 0.168,
                           0.599, 0.144, 0.222, 0.737, 0.563, 0.073, 0.719,
                           0.422, -0.071, 0.093, 0.5, 0.402, 0.273, 0.367,
                           0.346, -0.076, -0.108, 0.117, -0.17, 0.273, 0.602,
                           -0.143, -0.148, 0.441, 0.546, -0.144, -0.23, 0.099,
                           0.523, -0.13, 0.146, 0.549, 0.544, 0.579, -0.071,
                           0.245, 0.263, 0.223, -0.03, 0.476, 0.591, 0.585,
                           0.362, 0.4, 0.043, 0.262, 0.588, 0.722, 0.105, 0.029,
                           0.805, -0.018, 0.604, 0.682, 0.167, 0.045, -0.064,
                           -0.081, 0.506, -0.048, 0.677, -0.03, 0.076, 0.903]
        self.max_bounds = [1.583, 1.424, 1.516, 1.334, 1.433, 1.226, 1.580,
                           1.760, 1.530, 1.641, 1.209, 1.38, 1.783, 1.681,
                           1.461, 1.568, 1.794, 1.059, 1.603, 1.05, 1.415,
                           1.305, 1.874, 1.015, 1.768, 1.971, 1.99, 1.789,
                           1.439, 1.498, 1.214, 1.643, 1.32, 1.96, 1.727,
                           1.412, 1.745, 1.268, 1.44, 1.933, 1.683, 1.213,
                           1.839, 1.629, 1.134, 1.207, 1.607, 1.63, 1.37,
                           1.575, 1.451, 1.044, 1.027, 1.313, 1.013, 1.384,
                           1.683, 1.093, 1.035, 1.612, 1.609, 1.016, 1.016,
                           1.19, 1.587, 1.058, 1.368, 1.631, 1.718, 1.693,
                           1.084, 1.454, 1.442, 1.353, 1.154, 1.676, 1.699,
                           1.728, 1.478, 1.555, 1.121, 1.451, 1.716, 1.893,
                           1.273, 1.255, 1.866, 1.232, 1.805, 1.908, 1.232,
                           1.239, 1.05, 1.078, 1.641, 1.191, 1.844, 1.174,
                           1.171, 1.994]
        # truncate the above vectors to the number of used dimensions
        self.cut(num_variables)


    def get_optimal_solutions(self, max_number):
        solutions = []
        for i in range(max_number):
            phenome = copy.copy(self.o)
            phenome[0] = self.o[0] + float(i) / float(max_number - 1)
            phenome[0] = max(phenome[0], self.min_bounds[0])
            phenome[0] = min(phenome[0], self.max_bounds[0])
            solutions.append(Individual(phenome))
        return solutions



class S_ZDT4(ShiftedProblem):
    """A shifted ZDT4 problem."""

    def __init__(self, num_variables, is_orig_behavior=False, **kwargs):
        # the problem that is extended here
        self.problem = ZDT4(num_variables)
        self.is_orig_behavior = is_orig_behavior
        self.used_indices_per_dim = [[0]]
        if is_orig_behavior:
            self.used_indices_per_dim.append(list(range(1, num_variables)))
        else:
            self.used_indices_per_dim.append(list(range(num_variables)))
        self.max_objectives = [3.0, 1000.0]
        self.min_objectives = [1.0, 1.0]
        self.do_maximize = False
        self.is_deterministic = True
        # call superclass constructor
        ShiftedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=2,
                                **kwargs)
        self.default_reference_set_size = 300
        self.o = [0.957, 0.436, 2.092, 5.523, 5.686, 3.616, 1.646, 9.461, 0.881,
                  7.606, 4.401, 4.251, 5.182, 6.320, 9.136, 9.871, 7.308, 6.021,
                  1.941, 0.640, 0.581, 4.970, 4.677, 4.436, 3.997, 1.971, 0.071,
                  8.880, 9.464, 4.152, 1.318, 4.620, 9.296, 2.804, 9.034, 1.787,
                  5.197, 7.792, 5.364, 7.301, 0.953, 6.922, 5.955, 5.000, 1.437,
                  1.800, 2.796, 2.448, 0.499, 2.813, 3.784, 5.816, 7.544, 9.607,
                  0.634, 7.079, 6.864, 9.367, 2.498, 3.362, 5.484, 8.693, 2.720,
                  0.246, 1.878, 7.354, 4.399, 8.886, 1.394, 4.045, 7.694, 1.343,
                  4.430, 4.077, 1.512, 5.488, 7.547, 3.081, 7.321, 7.537, 3.430,
                  1.710, 9.287, 3.121, 5.341, 1.471, 5.165, 3.627, 7.946, 1.710,
                  9.013, 7.844, 9.240, 6.567, 4.996, 3.462, 1.847, 2.767, 9.231,
                  8.492]
        self.d = [0.099, 1.905, 2.486, 1.323, 0.823, 1.519, 1.737, 1.969, 2.072,
                  1.949, 1.812, 1.895, 0.571, 2.378, 1.079, 0.673, 1.300, 1.929,
                  2.052, 1.499, 2.282, 1.721, 0.675, 1.275, 1.282, 2.080, 1.178,
                  1.539, 2.319, 0.672, 1.243, 0.883, 0.939, 2.239, 1.249, 1.833,
                  1.154, 1.773, 1.743, 2.152, 2.445, 1.783, 0.753, 1.610, 1.248,
                  0.749, 0.703, 1.544, 2.203, 2.355, 1.373, 1.570, 1.330, 0.834,
                  1.183, 0.731, 1.142, 1.991, 2.101, 1.163, 1.817, 0.849, 1.631,
                  0.934, 1.672, 1.313, 1.488, 0.826, 1.907, 2.250, 0.676, 0.593,
                  1.953, 0.699, 1.340, 1.880, 0.690, 1.655, 1.804, 2.296, 1.826,
                  0.856, 1.924, 1.652, 1.501, 0.903, 1.852, 1.661, 2.351, 2.107,
                  1.819, 0.574, 0.803, 1.662, 2.390, 2.402, 1.007, 0.654, 1.845,
                  2.116]
        self.lambdas = [5.055, 2.625, 2.011, 3.779, 6.077, 3.291, 2.878, 2.540,
                        2.413, 2.566, 2.760, 2.639, 8.752, 2.103, 4.633, 7.434,
                        3.847, 2.592, 2.437, 3.336, 2.191, 2.905, 7.409, 3.922,
                        3.901, 2.404, 4.245, 3.249, 2.156, 7.441, 4.024, 5.665,
                        5.327, 2.233, 4.003, 2.727, 4.334, 2.820, 2.869, 2.323,
                        2.045, 2.804, 6.644, 3.105, 4.007, 6.676, 7.116, 3.238,
                        2.269, 2.123, 3.642, 3.185, 3.759, 5.997, 4.228, 6.837,
                        4.378, 2.512, 2.380, 4.299, 2.752, 5.893, 3.066, 5.353,
                        2.990, 3.808, 3.360, 6.055, 2.622, 2.222, 7.394, 8.426,
                        2.560, 7.155, 3.732, 2.660, 7.246, 3.022, 2.772, 2.178,
                        2.738, 5.842, 2.599, 3.026, 3.332, 5.538, 2.700, 3.010,
                        2.126, 2.374, 2.748, 8.707, 6.230, 3.008, 2.092, 2.081,
                        4.963, 7.649, 2.710, 2.363]
        self.min_bounds = [0.858, -6.469, -5.394, -0.8, -0.137, -2.903, -5.091,
                           2.492, -6.191, 0.657, -2.411, -2.644, -0.389, -1.058,
                           3.057, 4.198, 1.008, -0.908, -5.111, -5.859, -6.701,
                           -1.751, -0.998, -1.839, -2.285, -5.109, -6.107,
                           2.341, 2.145, -1.52, -4.925, -1.263, 3.357, -4.435,
                           2.785, -5.046, -0.957, 1.019, -1.379, 0.149, -6.492,
                           0.139, 0.202, -1.61, -4.811, -3.949, -2.907, -4.096,
                           -6.704, -4.542, -2.589, -0.754, 1.214, 3.773, -5.549,
                           1.348, 0.722, 2.376, -4.603, -2.801, -1.333, 2.844,
                           -3.911, -5.688, -4.794, 1.041, -2.089, 3.06, -5.513,
                           -3.205, 2.018, -4.25, -2.523, -1.622, -4.828, -1.392,
                           1.857, -3.574, 0.517, 0.241, -3.396, -4.146, 2.363,
                           -3.531, -1.16, -4.432, -1.687, -3.034, 0.595, -5.397,
                           2.194, 2.27, 3.437, -0.095, -2.394, -3.94, -4.16,
                           -2.887, 2.386, 1.376]
        self.max_bounds = [1.957, 5.436, 7.092, 10.523, 10.686, 8.616, 6.646,
                           14.461, 5.881, 12.606, 9.401, 9.251, 10.182, 11.32,
                           14.136, 14.871, 12.308, 11.021, 6.941, 5.64, 5.581,
                           9.97, 9.677, 9.436, 8.997, 6.971, 5.071, 13.88,
                           14.464, 9.152, 6.318, 9.62, 14.296, 7.804, 14.034,
                           6.787, 10.197, 12.792, 10.364, 12.301, 5.953, 11.922,
                           10.955, 10, 6.437, 6.8, 7.796, 7.448, 5.499, 7.813,
                           8.784, 10.816, 12.544, 14.607, 5.634, 12.079, 11.864,
                           14.367, 7.498, 8.362, 10.484, 13.693, 7.72, 5.246,
                           6.878, 12.354, 9.399, 13.886, 6.394, 9.045, 12.694,
                           6.343, 9.43, 9.077, 6.512, 10.488, 12.547, 8.081,
                           12.321, 12.537, 8.43, 6.71, 14.287, 8.121, 10.341,
                           6.471, 10.165, 8.627, 12.946, 6.71, 14.013, 12.844,
                           14.24, 11.567, 9.996, 8.462, 6.847, 7.767, 14.231,
                           13.492]
        # truncate the above vectors to the number of used dimensions
        self.cut(num_variables)


    def calc_penalties(self, z):
        """Calculate the penalties from the z values."""
        penalties = []
        if z[0] >= 0:
            penalties.append(0.0)
        else:
            penalties.append(abs(z[0]) / self.d[0])
        for i in range(1, len(z)):
            if z[i] >= -5:
                penalties.append(0.0)
            else:
                penalties.append(abs(z[i] + 5.0) / self.d[i])
        return penalties


    def calc_z_prime(self, z):
        """Calculate the z' values from the z values."""
        z_prime = []
        if z[0] >= 0:
            z_prime.append(z[0])
        else:
            z_prime.append(-self.lambdas[0] * z[0])
        for i in range(1, len(z)):
            if z[i] >= -5:
                z_prime.append(z[i])
            else:
                z_prime.append(-5.0 - self.lambdas[i] * (z[i] + 5.0))
        return z_prime


    def get_optimal_solutions(self, max_number):
        solutions = []
        for i in range(max_number):
            phenome = copy.copy(self.o)
            phenome[0] = self.o[0] + float(i) / float(max_number - 1)
            phenome[0] = max(phenome[0], self.min_bounds[0])
            phenome[0] = min(phenome[0], self.max_bounds[0])
            solutions.append(Individual(phenome))
        return solutions



class R_ZDT4(RotatedProblem):
    """A rotated ZDT4 problem."""

    def __init__(self, num_variables, is_orig_behavior=False, phenome_preprocessor=None, **kwargs):
        # the problem that is extended here
        self.problem = ZDT4(num_variables)
        self.num_variables = num_variables
        self.is_orig_behavior = is_orig_behavior
        self.used_indices_per_dim = [[0], list(range(num_variables))]
        self.default_reference_set_size = 300
        if num_variables == 10:
            self.rotation_matrix = np.array([[0.522, -0.230, 0.087, 0.806, 0.131, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.009, 0.648, -0.707, 0.229, 0.167, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.404, 0.391, 0.160, -0.036, -0.811, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [-0.455, -0.450, -0.448, 0.303, -0.546, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [-0.598, 0.415, 0.515, 0.452, -0.016, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            self.lambdas = [0.042, 0.483, 0.510, 0.390, 0.459, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.min_bounds = [-7.330, -9.750, -9.150, -5.100, -7.700, -5.000, -5.000, -5.000, -5.000, -5.000]
            self.max_bounds = [7.852, 9.520, 9.237, 5.906, 7.831, 5.000, 5.000, 5.000, 5.000, 5.000]
        elif num_variables == 30:
            self.rotation_matrix = np.array([[-0.087, 0.057, -0.403, 0.349, 0.114, -0.206, 0.014, 0.335, 0.341, 0.100, -0.504, -0.127, -0.230, -0.186, 0.230, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.124, 0.496, 0.007, -0.121, -0.168, 0.130, 0.270, -0.210, 0.222, 0.135, 0.232, -0.498, -0.013, -0.003, 0.439, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.047, -0.215, -0.117, -0.107, -0.152, 0.442, -0.229, -0.172, 0.155, -0.591, -0.076, 0.025, -0.048, -0.432, 0.239, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.220, -0.082, 0.078, -0.130, 0.530, -0.100, 0.123, -0.206, -0.071, 0.048, -0.004, -0.382, -0.213, -0.489, -0.368, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.418, -0.423, -0.261, -0.271, -0.219, -0.449, -0.044, -0.134, 0.263, 0.065, 0.304, -0.217, 0.120, -0.089, -0.020, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.371, -0.463, 0.229, 0.087, 0.312, -0.209, 0.059, -0.074, 0.027, 0.093, 0.049, 0.111, 0.191, 0.060, 0.612, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.034, -0.066, 0.148, 0.416, -0.178, -0.172, 0.005, 0.043, -0.172, -0.143, 0.454, 0.029, -0.682, -0.060, 0.101, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.398, -0.152, 0.434, -0.122, 0.043, 0.282, -0.437, 0.056, -0.073, 0.380, -0.198, -0.274, -0.208, 0.023, 0.178, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.010, -0.376, -0.311, -0.070, -0.099, 0.497, 0.508, 0.147, -0.156, 0.393, 0.086, 0.100, -0.133, -0.081, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.256, -0.066, 0.423, -0.245, -0.181, 0.003, 0.101, 0.511, 0.564, 0.003, 0.085, 0.087, -0.088, -0.093, -0.195, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.014, 0.257, -0.063, -0.320, -0.081, -0.222, -0.139, 0.395, -0.448, 0.115, 0.112, 0.173, 0.120, -0.510, 0.256, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.381, 0.121, 0.040, -0.462, 0.399, -0.062, 0.332, 0.020, 0.037, -0.303, -0.071, 0.246, -0.322, 0.217, 0.203, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.283, -0.162, -0.275, -0.214, 0.069, 0.033, -0.191, 0.421, -0.237, -0.264, 0.070, -0.481, -0.086, 0.429, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.366, 0.079, -0.309, -0.318, -0.022, -0.040, -0.398, -0.294, 0.171, 0.329, 0.009, 0.322, -0.398, 0.127, -0.006, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.171, 0.138, -0.187, 0.195, 0.512, 0.280, -0.264, 0.204, 0.266, 0.067, 0.556, 0.105, 0.175, -0.029, -0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]])
            self.lambdas = [0.011, 0.125, 0.128, 0.128, 0.115, 0.132, 0.151,
                            0.117, 0.128, 0.134, 0.120, 0.117, 0.118, 0.118,
                            0.124, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                            1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                            1.000, 1.000]
            self.min_bounds = [-15.56, -15.462, -14.803, -15.393, -14.827,
                               -14.8, -15.504, -14.415, -14.314, -14.655,
                               -12.032, -15.374, -14.209, -13.398, -13.147,
                               -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0,
                               -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]
            self.max_bounds = [15.472, 15.519, 14.4, 15.742, 14.942, 14.593,
                               15.518, 14.749, 14.656, 14.755, 11.527, 15.247,
                               13.979, 13.212, 13.378, 5.0, 5.0, 5.0, 5.0, 5.0,
                               5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        else:
            raise Exception("Unsupported number of parameters. Use 10 or 30.")
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        self.max_objectives = [4.0, 500.0]
        self.min_objectives = [1.0, 1.0]
        self.do_maximize = False
        self.is_deterministic = True
        # call superclass constructor
        RotatedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=2,
                                phenome_preprocessor=preprocessor,
                                **kwargs)


    def calc_penalties(self, z):
        """Calculate the penalties from the z values."""
        penalties = []
        if z[0] < 0:
            penalties.append(-z[0])
        elif z[0] <= 1:
            penalties.append(0.0)
        else:
            penalties.append(z[0] - 1.0)
        for i in range(1, len(z)):
            if z[i] < -5:
                penalties.append(-5.0 - z[i])
            elif z[i] <= 5:
                penalties.append(0.0)
            else:
                penalties.append(z[i] - 5.0)
        return penalties


    def calc_z_prime(self, z):
        """Calculate the z' values from the z values."""
        z_prime = []
        if z[0] < 0:
            z_prime.append(-self.lambdas[0] * z[0])
        elif z[0] <= 1:
            z_prime.append(z[0])
        else:
            if self.is_orig_behavior:
                z_prime.append(1.0 - self.lambdas[0] * (z[0] - 1.0))
            else:
                z_prime.append(self.lambdas[0] * z[0])
        for i in range(1, len(z)):
            if z[i] < -5:
                z_prime.append(-5.0 - self.lambdas[i] * (z[i] + 5.0))
            elif z[i] <= 5:
                z_prime.append(z[i])
            else:
                z_prime.append(5.0 - self.lambdas[i] * (z[i] - 5.0))
        return z_prime


    def get_optimal_solutions(self, number):
        raise NotImplementedError("The Pareto-set is unknown because of rotation, but you can still obtain the Pareto-front.")


    def sample_pareto_front(self, num_points, oversampling_factor=100):
        # the front is identical to the one of the shifted problem
        dummy_problem = S_ZDT4(self.num_variables, self.is_orig_behavior)
        samples = dummy_problem.sample_pareto_front(num_points, oversampling_factor)
        # erase genome because the values don't match the objective values
        # of the rotated problem
        for sample in samples:
            sample.phenome = None
        return samples



class S_ZDT6(ShiftedProblem):
    """A shifted ZDT6 problem."""

    def __init__(self, num_variables, is_orig_behavior=False, **kwargs):
        # the problem that is extended here
        self.problem = ZDT6(num_variables)
        self.is_orig_behavior = is_orig_behavior
        self.used_indices_per_dim = [[0]]
        if is_orig_behavior:
            self.used_indices_per_dim.append(list(range(1, num_variables)))
        else:
            self.used_indices_per_dim.append(list(range(num_variables)))
        self.max_objectives = [3.0, 20.0]
        self.min_objectives = [1.0, 1.0]
        self.do_maximize = False
        self.is_deterministic = True
        # call superclass constructor
        ShiftedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=2,
                                **kwargs)
        self.default_reference_set_size = 300
        self.o = [0.360, 0.549, 0.262, 0.597, 0.049, 0.571, 0.701, 0.962, 0.751,
                  0.740, 0.432, 0.634, 0.803, 0.084, 0.945, 0.916, 0.602, 0.254,
                  0.873, 0.513, 0.733, 0.422, 0.961, 0.072, 0.553, 0.292, 0.858,
                  0.336, 0.680, 0.053, 0.357, 0.498, 0.434, 0.562, 0.617, 0.113,
                  0.898, 0.755, 0.791, 0.815, 0.670, 0.201, 0.273, 0.626, 0.537,
                  0.060, 0.089, 0.271, 0.409, 0.474, 0.909, 0.596, 0.329, 0.478,
                  0.597, 0.161, 0.829, 0.956, 0.596, 0.029, 0.812, 0.610, 0.701,
                  0.092, 0.425, 0.376, 0.166, 0.833, 0.839, 0.452, 0.957, 0.147,
                  0.870, 0.769, 0.444, 0.621, 0.952, 0.640, 0.247, 0.353, 0.188,
                  0.491, 0.409, 0.464, 0.611, 0.071, 0.314, 0.608, 0.175, 0.621,
                  0.246, 0.587, 0.506, 0.465, 0.541, 0.942, 0.342, 0.402, 0.308,
                  0.412]
        self.d = [0.155, 0.119, 0.185, 0.064, 0.070, 0.203, 0.166, 0.151, 0.219,
                  0.083, 0.161, 0.057, 0.169, 0.072, 0.135, 0.114, 0.241, 0.129,
                  0.230, 0.181, 0.195, 0.058, 0.152, 0.107, 0.119, 0.219, 0.203,
                  0.212, 0.126, 0.238, 0.164, 0.236, 0.086, 0.087, 0.188, 0.244,
                  0.146, 0.124, 0.218, 0.196, 0.120, 0.140, 0.120, 0.207, 0.205,
                  0.114, 0.107, 0.228, 0.097, 0.208, 0.105, 0.120, 0.135, 0.196,
                  0.183, 0.111, 0.081, 0.236, 0.183, 0.171, 0.063, 0.160, 0.246,
                  0.091, 0.064, 0.188, 0.222, 0.082, 0.174, 0.114, 0.155, 0.209,
                  0.179, 0.130, 0.184, 0.200, 0.108, 0.143, 0.116, 0.155, 0.078,
                  0.189, 0.128, 0.171, 0.168, 0.226, 0.061, 0.250, 0.201, 0.226,
                  0.065, 0.194, 0.114, 0.159, 0.135, 0.239, 0.167, 0.204, 0.095,
                  0.091]
        self.lambdas = [3.236, 4.201, 2.701, 7.775, 7.148, 2.465, 3.020, 3.322,
                        2.285, 6.004, 3.106, 8.801, 2.956, 6.918, 3.708, 4.403,
                        2.077, 3.884, 2.171, 2.760, 2.567, 8.636, 3.299, 4.666,
                        4.208, 2.285, 2.469, 2.362, 3.963, 2.099, 3.052, 2.120,
                        5.809, 5.753, 2.654, 2.053, 3.427, 4.046, 2.297, 2.548,
                        4.151, 3.577, 4.165, 2.417, 2.445, 4.405, 4.668, 2.192,
                        5.147, 2.398, 4.743, 4.157, 3.710, 2.553, 2.733, 4.519,
                        6.159, 2.121, 2.738, 2.926, 7.918, 3.130, 2.034, 5.495,
                        7.846, 2.666, 2.253, 6.068, 2.881, 4.388, 3.220, 2.389,
                        2.791, 3.832, 2.711, 2.496, 4.624, 3.505, 4.297, 3.223,
                        6.428, 2.644, 3.920, 2.917, 2.978, 2.215, 8.138, 2.000,
                        2.490, 2.215, 7.711, 2.574, 4.395, 3.135, 3.705, 2.091,
                        3.003, 2.456, 5.241, 5.499]
        self.min_bounds = [0.205, 0.43, 0.077, 0.533, -0.021, 0.368, 0.535,
                           0.811, 0.532, 0.657, 0.271, 0.577, 0.634, 0.012,
                           0.81, 0.802, 0.361, 0.125, 0.643, 0.332, 0.538,
                           0.364, 0.809, -0.035, 0.434, 0.073, 0.655, 0.124,
                           0.554, -0.185, 0.193, 0.262, 0.348, 0.475, 0.429,
                           -0.131, 0.752, 0.631, 0.573, 0.619, 0.55, 0.061,
                           0.153, 0.419, 0.332, -0.054, -0.018, 0.043, 0.312,
                           0.266, 0.804, 0.476, 0.194, 0.282, 0.414, 0.05,
                           0.748, 0.72, 0.413, -0.142, 0.749, 0.45, 0.455,
                           0.001, 0.361, 0.188, -0.056, 0.751, 0.665, 0.338,
                           0.802, -0.062, 0.691, 0.639, 0.26, 0.421, 0.844,
                           0.497, 0.131, 0.198, 0.11, 0.302, 0.281, 0.293,
                           0.443, -0.155, 0.253, 0.358, -0.026, 0.395, 0.181,
                           0.393, 0.392, 0.306, 0.406, 0.703, 0.175, 0.198,
                           0.213, 0.321]
        self.max_bounds = [1.36, 1.549, 1.262, 1.597, 1.049, 1.571, 1.701,
                           1.962, 1.751, 1.74, 1.432, 1.634, 1.803, 1.084,
                           1.945, 1.916, 1.602, 1.254, 1.873, 1.513, 1.733,
                           1.422, 1.961, 1.072, 1.553, 1.292, 1.858, 1.336,
                           1.68, 1.053, 1.357, 1.498, 1.434, 1.562, 1.617,
                           1.113, 1.898, 1.755, 1.791, 1.815, 1.67, 1.201,
                           1.273, 1.626, 1.537, 1.06, 1.089, 1.271, 1.409,
                           1.474, 1.909, 1.596, 1.329, 1.478, 1.597, 1.161,
                           1.829, 1.956, 1.596, 1.029, 1.812, 1.61, 1.701,
                           1.092, 1.425, 1.376, 1.166, 1.833, 1.839, 1.452,
                           1.957, 1.147, 1.87, 1.769, 1.444, 1.621, 1.952,
                           1.64, 1.247, 1.353, 1.188, 1.491, 1.409, 1.464,
                           1.611, 1.071, 1.314, 1.608, 1.175, 1.621, 1.246,
                           1.587, 1.506, 1.465, 1.541, 1.942, 1.342, 1.402,
                           1.308, 1.412]
        # truncate the above vectors to the number of used dimensions
        self.cut(num_variables)


    def get_optimal_solutions(self, max_number):
        solutions = []
        for i in range(max_number):
            phenome = copy.copy(self.o)
            phenome[0] = self.o[0] + float(i) / float(max_number - 1)
            phenome[0] = max(phenome[0], self.min_bounds[0])
            phenome[0] = min(phenome[0], self.max_bounds[0])
            solutions.append(Individual(phenome))
        return solutions



class S_DTLZ2(ShiftedProblem):
    """A shifted DTLZ2 problem."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        self.max_objectives = [10.0] * num_objectives
        self.min_objectives = [1.0] * num_objectives
        self.do_maximize = False
        self.is_deterministic = True
        # call superclass constructor
        ShiftedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=num_objectives,
                                **kwargs)
        # the problem that is extended here
        self.problem = DTLZ2(num_objectives, num_variables)
        self.o = [0.366, 0.303, 0.852, 0.759, 0.950, 0.558, 0.014, 0.596, 0.816,
                  0.977, 0.222, 0.704, 0.522, 0.933, 0.713, 0.228, 0.450, 0.172,
                  0.969, 0.356, 0.049, 0.755, 0.895, 0.286, 0.251, 0.933, 0.131,
                  0.941, 0.702, 0.848]
        self.d = [0.155, 0.119, 0.185, 0.064, 0.07, 0.203, 0.166, 0.151, 0.219,
                  0.083, 0.161, 0.057, 0.169, 0.072, 0.135, 0.114, 0.241, 0.129,
                  0.23, 0.181, 0.195, 0.058, 0.152, 0.107, 0.119, 0.219, 0.203,
                  0.212, 0.126, 0.238]
        self.lambdas = [3.236, 4.201, 2.701, 7.775, 7.148, 2.465, 3.02, 3.322,
                        2.285, 6.004, 3.106, 8.801, 2.956, 6.918, 3.708, 4.403,
                        2.077, 3.884, 2.171, 2.76, 2.567, 8.636, 3.299, 4.666,
                        4.208, 2.285, 2.469, 2.362, 3.963, 2.099]
        self.min_bounds = [0.211, 0.184, 0.667, 0.695, 0.88, 0.355, -0.152,
                           0.445, 0.597, 0.894, 0.061, 0.647, 0.353, 0.861,
                           0.578, 0.114, 0.209, 0.043, 0.739, 0.175, -0.146,
                           0.697, 0.743, 0.179, 0.132, 0.714, -0.072, 0.729,
                           0.576, 0.610]
        self.max_bounds = [1.366, 1.303, 1.852, 1.759, 1.95, 1.558, 1.014,
                           1.596, 1.816, 1.977, 1.222, 1.704, 1.522, 1.933,
                           1.713, 1.228, 1.45, 1.172, 1.969, 1.356, 1.049,
                           1.755, 1.895, 1.286, 1.251, 1.933, 1.131, 1.941,
                           1.702, 1.848]
        # truncate the above vectors to the number of used dimensions
        self.cut(num_variables)
        # add objective functions
        k = num_variables - num_objectives + 1
        used_indices_per_dim = []
        for i in range(1, num_objectives + 1):
            # determine which variables are used by the original function
            used_indices = list(range(num_variables - k, num_variables))
            for j in range(1, num_objectives - i + 1):
                used_indices.append(j - 1)
            if i > 1:
                index = (num_objectives - i + 1) - 1
                used_indices.append(index)
            used_indices_per_dim.append(used_indices)
        self.used_indices_per_dim = used_indices_per_dim
        if num_objectives <= 4:
            self.default_reference_set_size = 500
        else:
            self.default_reference_set_size = 1000


    def get_optimal_solutions(self, max_number):
        solutions = self.problem.get_optimal_solutions(max_number)
        for solution in solutions:
            for i in range(len(solution.phenome)):
                solution.phenome[i] += self.o[i]
        return solutions



class R_DTLZ2(RotatedProblem):
    """A rotated DTLZ2 problem.

    Note that this problem, as it was used in the CEC 2007 contest, was
    based on DTLZ3 and not DTLZ2. Thus, if you set `is_orig_behavior` to
    True, it uses DTLZ3, else DTLZ2.

    """
    def __init__(self, num_objectives, num_variables, is_orig_behavior=False, phenome_preprocessor=None, **kwargs):
        self.max_objectives = [10.0] * num_objectives
        if is_orig_behavior:
            self.max_objectives = [8000.0] * num_objectives
        self.min_objectives = [1.0] * num_objectives
        self.do_maximize = False
        self.is_deterministic = True
        self.is_orig_behavior = is_orig_behavior
        # the problem that is extended here
        if is_orig_behavior:
            # in the original implementation the problem is actually based on DTLZ3
            self.problem = DTLZ3(num_objectives, num_variables)
        else:
            self.problem = DTLZ2(num_objectives, num_variables)
        self.num_variables = num_variables
        if num_variables == 30:
            self.rotation_matrix = np.array([[-0.376, 0.392, -0.034, 0.074, -0.124, -0.013, -0.430, 0.168, 0.144, 0.334, 0.054, -0.486, 0.255, -0.081, 0.163, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.202, -0.401, 0.426, 0.136, 0.123, -0.437, -0.297, -0.182, 0.417, 0.119, 0.150, 0.190, -0.129, -0.026, 0.085, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.218, -0.035, 0.074, -0.107, -0.412, -0.093, 0.659, 0.181, 0.291, 0.148, -0.102, -0.082, -0.185, -0.200, 0.300, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.053, -0.211, -0.173, -0.130, 0.579, 0.034, 0.065, 0.171, 0.132, -0.143, -0.081, -0.311, -0.039, 0.279, 0.562, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.104, 0.331, -0.031, -0.347, 0.036, -0.345, 0.077, 0.236, 0.130, -0.088, 0.323, -0.020, -0.280, 0.503, -0.340, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.219, -0.212, 0.294, -0.514, -0.332, 0.373, -0.264, 0.050, -0.058, -0.428, 0.134, 0.006, 0.062, 0.047, 0.155, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.165, 0.324, 0.386, -0.425, 0.366, -0.075, 0.033, -0.112, -0.335, 0.261, -0.344, 0.189, -0.100, -0.176, 0.096, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.135, 0.328, -0.066, 0.268, 0.062, 0.093, 0.011, -0.154, -0.203, -0.167, 0.562, 0.202, -0.360, -0.181, 0.417, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.238, -0.064, 0.093, 0.111, -0.143, -0.131, -0.303, 0.296, -0.185, -0.138, -0.272, -0.317, -0.655, -0.206, -0.041, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.623, 0.161, 0.259, -0.132, -0.110, 0.340, -0.070, -0.132, 0.288, 0.413, 0.165, -0.020, -0.137, 0.175, 0.146, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.008, -0.130, -0.578, -0.273, -0.154, -0.064, -0.313, 0.199, 0.028, 0.305, -0.087, 0.504, -0.119, -0.032, 0.195, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [-0.346, -0.347, 0.078, 0.209, -0.024, 0.315, 0.083, 0.056, -0.381, 0.452, 0.069, -0.061, -0.231, 0.425, -0.093, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.168, -0.274, -0.170, -0.349, -0.098, -0.361, 0.059, -0.355, -0.355, 0.190, 0.351, -0.392, 0.068, -0.178, 0.031, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.259, 0.026, 0.291, 0.178, -0.206, -0.358, 0.011, 0.419, -0.374, -0.001, 0.059, 0.182, 0.363, 0.237, 0.326, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.064, -0.173, 0.104, -0.088, 0.327, 0.178, 0.075, 0.577, 0.031, 0.149, 0.402, 0.035, 0.096, -0.463, -0.248, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000],
                                             [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]])
            self.lambdas = [0.113, 0.105, 0.117, 0.119, 0.108, 0.110, 0.101,
                            0.107, 0.111, 0.109, 0.120, 0.108, 0.101, 0.105,
                            0.116, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                            1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                            1.000, 1.000]
            self.min_bounds = [-1.773, -1.846, -1.053, -2.370, -1.603, -1.878,
                               -1.677, -0.935, -1.891, -0.964, -0.885, -1.690,
                               -2.235, -1.541, -0.720, 0.000, 0.000, 0.000,
                               0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                               0.000, 0.000, 0.000, 0.000, 0.000]
            self.max_bounds = [1.403, 1.562, 2.009, 0.976, 1.490, 1.334, 1.074,
                               2.354, 1.462, 2.372, 2.267, 1.309, 0.842, 1.665,
                               2.476, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                               1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                               1.000, 1.000]
        elif num_variables == 10:
            self.rotation_matrix = np.array([[-0.444, -0.380, -0.510, 0.124, 0.619, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.214, -0.570, -0.445, 0.239, -0.612, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [-0.675, 0.462, -0.336, -0.093, -0.458, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.526, 0.376, -0.644, -0.379, 0.154, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.160, 0.419, -0.120, 0.880, 0.097, 0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
            self.lambdas = [0.313, 0.312, 0.321, 0.316, 0.456, 1.0, 1.0, 1.0, 1.0, 1.0]
            self.min_bounds = [-1.118, -0.951, -2.055, -0.472, -1.070, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.max_bounds = [0.899, 1.257, 0.0, 1.244, 0.869, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            raise Exception("Unsupported number of parameters. Use 10 or 30.")
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        # call superclass constructor
        RotatedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=num_objectives,
                                phenome_preprocessor=preprocessor,
                                **kwargs)
        k = num_variables - num_objectives + 1
        used_indices_per_dim = []
        for i in range(1, num_objectives + 1):
            # determine which variables are used by the original function
            used_indices = list(range(num_variables - k, num_variables))
            for j in range(1, num_objectives - i + 1):
                used_indices.append(j - 1)
            if i > 1:
                index = (num_objectives - i + 1) - 1
                used_indices.append(index)
            used_indices_per_dim.append(used_indices)
        self.used_indices_per_dim = used_indices_per_dim
        if num_objectives <= 4:
            self.default_reference_set_size = 500
        else:
            self.default_reference_set_size = 1000


    def calc_penalties(self, z):
        """Calculate the penalties from the z values."""
        penalties = []
        for i in range(0, len(z)):
            if z[i] < 0:
                penalties.append(-z[i])
            elif z[i] <= 1:
                penalties.append(0.0)
            else:
                penalties.append(z[i] - 1.0)
        return penalties


    def calc_z_prime(self, z):
        """Calculate the z' values from the z values."""
        z_prime = []
        for i in range(0, len(z)):
            if z[i] < 0:
                z_prime.append(-self.lambdas[i] * z[i])
            elif z[i] <= 1:
                z_prime.append(z[i])
            else:
                if self.is_orig_behavior:
                    z_prime.append(1 - self.lambdas[i] * (z[i] - 1))
                else:
                    z_prime.append(self.lambdas[i] * z[i])
        return z_prime


    def get_optimal_solutions(self, number):
        raise NotImplementedError("The Pareto-set is unknown because of rotation, but you can still obtain the Pareto-front.")


    def sample_pareto_front(self, num_points):
        # the front is identical to the one of the shifted problem
        if self.is_orig_behavior:
            dummy_problem = S_DTLZ3(self.problem.num_objectives,
                                    self.problem.num_variables)
        else:
            dummy_problem = S_DTLZ2(self.problem.num_objectives,
                                    self.problem.num_variables)
        samples = dummy_problem.sample_pareto_front(num_points)
        # erase genome because the values don't match the context of the rotated problem
        for sample in samples:
            sample.phenome = None
        return samples



class S_DTLZ3(ShiftedProblem):
    """A shifted DTLZ3 problem."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        self.max_objectives = [6000.0] * num_objectives
        self.min_objectives = [1.0] * num_objectives
        self.do_maximize = False
        self.is_deterministic = True
        # call superclass constructor
        ShiftedProblem.__init__(self,
                                self.objective_function,
                                num_objectives=num_objectives,
                                **kwargs)
        # the problem that is extended here
        self.problem = DTLZ3(num_objectives, num_variables)
        self.o = [0.209, 0.455, 0.081, 0.851, 0.562, 0.319, 0.375, 0.868, 0.372,
                  0.074, 0.200, 0.049, 0.567, 0.122, 0.522, 0.117, 0.770, 0.375,
                  0.823, 0.047, 0.598, 0.949, 0.289, 0.889, 0.102, 0.065, 0.234,
                  0.933, 0.063, 0.264]
        self.d = [0.155, 0.119, 0.185, 0.064, 0.07, 0.203, 0.166, 0.151, 0.219,
                  0.083, 0.161, 0.057, 0.169, 0.072, 0.135, 0.114, 0.241, 0.129,
                  0.23, 0.181, 0.195, 0.058, 0.152, 0.107, 0.119, 0.219, 0.203,
                  0.212, 0.126, 0.238]
        self.lambdas = [3.236, 4.201, 2.701, 7.775, 7.148, 2.465, 3.02, 3.322,
                        2.285, 6.004, 3.106, 8.801, 2.956, 6.918, 3.708, 4.403,
                        2.077, 3.884, 2.171, 2.76, 2.567, 8.636, 3.299, 4.666,
                        4.208, 2.285, 2.469, 2.362, 3.963, 2.099]
        self.min_bounds = [0.054, 0.336, -0.104, 0.787, 0.492, 0.116, 0.209,
                           0.717, 0.153, -0.009, 0.039, -0.008, 0.398, 0.05,
                           0.387, 0.003, 0.529, 0.246, 0.593, -0.134, 0.403,
                           0.891, 0.137, 0.782, -0.017, -0.154, 0.031, 0.721,
                           -0.063, 0.026]
        self.max_bounds = [1.209, 1.455, 1.081, 1.851, 1.562, 1.319, 1.375,
                           1.868, 1.372, 1.074, 1.2, 1.049, 1.567, 1.122, 1.522,
                           1.117, 1.77, 1.375, 1.823, 1.047, 1.598, 1.949,
                           1.289, 1.889, 1.102, 1.065, 1.234, 1.933, 1.063,
                           1.264]
        # truncate the above vectors to the number of used dimensions
        self.cut(num_variables)
        # add objective functions
        k = num_variables - num_objectives + 1
        used_indices_per_dim = []
        for i in range(1, num_objectives + 1):
            # determine which variables are used by the original function
            used_indices = list(range(num_variables - k, num_variables))
            for j in range(1, num_objectives - i + 1):
                used_indices.append(j - 1)
            if i > 1:
                index = (num_objectives - i + 1) - 1
                used_indices.append(index)
            used_indices_per_dim.append(used_indices)
        self.used_indices_per_dim = used_indices_per_dim
        if num_objectives <= 4:
            self.default_reference_set_size = 500
        else:
            self.default_reference_set_size = 1000


    def get_optimal_solutions(self, max_number):
        solutions = self.problem.get_optimal_solutions(max_number)
        for solution in solutions:
            for i in range(len(solution.phenome)):
                solution.phenome[i] += self.o[i]
        return solutions



class CEC2007(list):
    """The CEC 2007 problem collection.

    The collection was defined in [Huang2007]_. This class inherits from
    :class:`list` and fills itself with the following problems: OKA2,
    SYMPART, S_ZDT1, S_ZDT2, S_ZDT4, R_ZDT4, S_ZDT6, 3-D S_DTLZ2,
    5-D S_DTLZ2, 3-D R_DTLZ2, 5-D R_DTLZ2, 3-D S_DTLZ3, 5-D S_DTLZ3,
    3-D WFG1, 5-D WFG1, 3-D WFG8, 5-D WFG8, 3-D WFG9, 5-D WFG9.

    """
    def __init__(self, is_orig_behavior=False, **kwargs):
        """Constructor.

        Parameters
        ----------
        is_orig_behavior : bool, optional
            A flag indicating if the behavior of the C implementation
            should be used. Default is False.
        kwargs
            Arbitrary keyword arguments, passed through to the constructors
            of the single problems.

        """
        oka2 = OKA2(is_orig_behavior, **kwargs)
        sympart = SYMPART(30, **kwargs)
        # ZDT derived problems
        s_zdt1 = S_ZDT1(30, is_orig_behavior, **kwargs)
        s_zdt2 = S_ZDT2(30, is_orig_behavior, **kwargs)
        s_zdt4 = S_ZDT4(30, is_orig_behavior, **kwargs)
        r_zdt4 = R_ZDT4(10, is_orig_behavior, **kwargs)
        s_zdt6 = S_ZDT6(30, is_orig_behavior, **kwargs)
        # DTLZ derived problems
        s_dtlz2_3 = S_DTLZ2(3, 30, **kwargs)
        s_dtlz2_5 = S_DTLZ2(5, 30, **kwargs)
        r_dtlz2_3 = R_DTLZ2(3, 30, is_orig_behavior, **kwargs)
        r_dtlz2_5 = R_DTLZ2(5, 30, is_orig_behavior, **kwargs)
        s_dtlz3_3 = S_DTLZ3(3, 30, **kwargs)
        s_dtlz3_5 = S_DTLZ3(5, 30, **kwargs)
        # WFG problems
        wfg1_3 = WFG1(3, 24, 4, **kwargs)
        wfg1_5 = WFG1(5, 28, 8, **kwargs)
        wfg8_3 = WFG8(3, 24, 4, **kwargs)
        wfg8_5 = WFG8(5, 28, 8, **kwargs)
        wfg9_3 = WFG9(3, 24, 4, **kwargs)
        wfg9_5 = WFG9(5, 28, 8, **kwargs)
        # create test suite
        list.__init__(self, [oka2, sympart, s_zdt1, s_zdt2, s_zdt4, r_zdt4,
                             s_zdt6, s_dtlz2_3, s_dtlz2_5, r_dtlz2_3, r_dtlz2_5,
                             s_dtlz3_3, s_dtlz3_5, wfg1_3, wfg1_5, wfg8_3,
                             wfg8_5, wfg9_3, wfg9_5])
