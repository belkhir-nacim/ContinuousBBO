"""
This module contains the ZDT suite for multiobjective optimization as defined
in [Zitzler2000]_. The problems have two objectives each. Except for ZDT5,
the search spaces are continuous.
"""

import math

from benchmark.base import BoundConstraintsChecker, Individual
from benchmark.multiobjective import MultiObjectiveTestProblem

__all__ = ["ZDT", "ZDTBaseProblem", "ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT5",
           "ZDT6"]

class ZDT_f2:
    """The f2 function for all ZDT problems."""

    def __init__(self, f1, g, h):
        self.f1 = f1
        self.g = g
        self.h = h


    def __call__(self, phenome):
        g_value = self.g(phenome)
        f1_value = self.f1(phenome)
        return g_value * self.h(f1_value, g_value)



def ZDT1to4_f1(phenome):
    """The f1 function for ZDT1, ZDT2, ZDT3 and ZDT4."""
    return phenome[0]



class ZDT1to3_g:
    """The g function for ZDT1, ZDT2 and ZDT3."""

    def __init__(self, num_variables):
        self.num_variables = num_variables


    def __call__(self, phenome):
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(1, n):
            temp_sum += phenome[i] / float(n - 1)
        return 1.0 + (9.0 * temp_sum)



class ZDTBaseProblem(MultiObjectiveTestProblem):

    def get_optimal_solutions(self, max_number=100):
        """Return Pareto-optimal solutions.

        .. note:: The returned solutions do not yet contain the objective
            values.

        Parameters
        ----------
        max_number : int, optional
            As the number of Pareto-optimal solutions is infinite, the
            returned set has to be restricted to a finite sample.

        Returns
        -------
        solutions : list of Individual
            The Pareto-optimal solutions

        """
        assert max_number > 1
        solutions = []
        for i in range(max_number):
            phenome = [0.0] * self.num_variables
            phenome[0] = float(i) / float(max_number - 1)
            solutions.append(Individual(phenome))
        return solutions



class ZDT1(ZDTBaseProblem):
    """The ZDT1 problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        f2 = ZDT_f2(ZDT1to4_f1, ZDT1to3_g(num_variables), self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        ZDTBaseProblem.__init__(self,
                                [ZDT1to4_f1, f2],
                                num_objectives=2,
                                phenome_preprocessor=preprocessor,
                                **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables


    @staticmethod
    def h(f1_value, g_value):
        """The h function of ZDT1."""
        return 1.0 - math.sqrt(f1_value / g_value)



class ZDT2(ZDTBaseProblem):
    """The ZDT2 problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        f2 = ZDT_f2(ZDT1to4_f1, ZDT1to3_g(num_variables), self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        ZDTBaseProblem.__init__(self,
                                [ZDT1to4_f1, f2],
                                num_objectives=2,
                                phenome_preprocessor=preprocessor,
                                **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables


    @staticmethod
    def h(f1_value, g_value):
        """The h function of ZDT2."""
        return 1.0 - (f1_value / g_value) ** 2



class ZDT3(ZDTBaseProblem):
    """The ZDT3 problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        f2 = ZDT_f2(ZDT1to4_f1, ZDT1to3_g(num_variables), self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        ZDTBaseProblem.__init__(self,
                                [ZDT1to4_f1, f2],
                                num_objectives=2,
                                phenome_preprocessor=preprocessor,
                                **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables


    @staticmethod
    def h(f1_value, g_value):
        """The h function of ZDT3."""
        fraction = f1_value / g_value
        return_value = 1.0 - math.sqrt(fraction)
        return_value -= fraction * math.sin(10.0 * math.pi * f1_value)
        return return_value



class ZDT4(ZDTBaseProblem):
    """The ZDT4 problem."""

    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        f2 = ZDT_f2(ZDT1to4_f1, self.g, self.h)
        self.min_bounds = [-5.0] * num_variables
        self.min_bounds[0] = 0.0
        self.max_bounds = [5.0] * num_variables
        self.max_bounds[0] = 1.0
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        ZDTBaseProblem.__init__(self,
                                [ZDT1to4_f1, f2],
                                num_objectives=2,
                                phenome_preprocessor=preprocessor,
                                **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables


    @staticmethod
    def h(f1_value, g_value):
        """The h function of ZDT4."""
        return 1.0 - math.sqrt(f1_value / g_value)


    def g(self, phenome):
        """The g function of ZDT4."""
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        four_pi = 4 * math.pi
        for i in range(1, n):
            x = phenome[i]
            temp_sum += x ** 2 - 10.0 * math.cos(four_pi * x)
        return 1.0 + 10.0 * (n - 1) + temp_sum



class ZDT5(ZDTBaseProblem):
    """ZDT5 is the only problem with binary encoding in this suite."""

    def __init__(self, **kwargs):
        f2 = ZDT_f2(self.f1, self.g, self.h)
        ZDTBaseProblem.__init__(self, [self.f1, f2], num_objectives=2, **kwargs)
        self.is_deterministic = True
        self.do_maximize = False


    @staticmethod
    def h(f1_value, g_value):
        """The h function of ZDT5."""
        return 1.0 / f1_value


    @staticmethod
    def u(phenome):
        for phene in phenome:
            assert phene in (0, 1)
        return sum(phenome)


    @staticmethod
    def v(u_value):
        if u_value < 5:
            return 2 + u_value
        else:
            return 1


    def f1(self, phenome):
        """The f1 function of ZDT5."""
        bitvector = phenome[0]
        assert len(bitvector) == 30
        return 1 + self.u(bitvector)


    def g(self, phenome):
        """The g function of ZDT5."""
        n = len(phenome)
        assert n == 11
        temp_sum = 0
        for i in range(1, n):
            bitvector = phenome[i]
            assert len(bitvector) == 5
            temp_sum += self.v(self.u(bitvector))
        return temp_sum


    def get_optimal_solutions(self, max_number=None):
        raise NotImplementedError("Optimal solutions are unknown.")



class ZDT6(ZDTBaseProblem):
    """The ZDT6 problem."""

    def __init__(self, num_variables=10, phenome_preprocessor=None, **kwargs):
        f2 = ZDT_f2(self.f1, self.g, self.h)
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        ZDTBaseProblem.__init__(self,
                                [self.f1, f2],
                                num_objectives=2,
                                phenome_preprocessor=preprocessor,
                                **kwargs)
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables


    @staticmethod
    def h(f1_value, g_value):
        """The h function of ZDT6."""
        return 1.0 - (f1_value / g_value) ** 2


    @staticmethod
    def f1(phenome):
        """The f1 function of ZDT6."""
        x = phenome[0]
        return 1.0 - math.exp(-4.0 * x) * math.pow(math.sin(6.0 * math.pi * x), 6.0)


    def g(self, phenome):
        """The g function of ZDT6."""
        n = len(phenome)
        assert n == self.num_variables
        temp_sum = 0.0
        for i in range(1, n):
            temp_sum += phenome[i]
        return 1.0 + (9.0 * math.pow(temp_sum / (n - 1), 0.25))



class ZDT(list):
    """The whole collection.

     This class inherits from :class:`list` and by default generates all
     problems with their default configuration.

    Parameters
    ----------
    kwargs
        Arbitrary keyword arguments, passed through to the constructors of
        the ZDT problems.

    References
    ----------
    .. [Zitzler2000] Zitzler, E., Deb, K., and Thiele, L. (2000).
        Comparison of Multiobjective Evolutionary Algorithms: Empirical
        Results. Evolutionary Computation 8(2).

    """
    def __init__(self, **kwargs):
        list.__init__(self, [ZDT1(**kwargs),
                             ZDT2(**kwargs),
                             ZDT3(**kwargs),
                             ZDT4(**kwargs),
                             ZDT5(**kwargs),
                             ZDT6(**kwargs)])
