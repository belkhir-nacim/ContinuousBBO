"""
This module contains the DTLZ problem collection.

Note that this collection is defined differently in two papers. The
technical report [Deb2001]_ contains nine problems, while [Deb2002]_
contains only seven. Additionally, the numbering differs from the fifth
problem on. Here, the first seven problems from [Deb2001]_ are implemented,
just as in the `PISA library
<http://www.tik.ee.ethz.ch/pisa/variators/dtlz/?page=dtlz.php>`_.

=============================== ====================== ======================
Class name                      Name in [Deb2001]_     Name in [Deb2002]_
=============================== ====================== ======================
:class:`benchmark.dtlz.DTLZ1` DTLZ1                  DTLZ1
:class:`benchmark.dtlz.DTLZ2` DTLZ2                  DTLZ2
:class:`benchmark.dtlz.DTLZ3` DTLZ3                  DTLZ3
:class:`benchmark.dtlz.DTLZ4` DTLZ4                  DTLZ4
:class:`benchmark.dtlz.DTLZ5` DTLZ5                  *missing*
:class:`benchmark.dtlz.DTLZ6` DTLZ6                  DTLZ5
:class:`benchmark.dtlz.DTLZ7` DTLZ7                  DTLZ6
*missing*                       DTLZ8                  DTLZ7
*missing*                       DTLZ9                  *missing*
=============================== ====================== ======================

References
----------
.. [Deb2001] Deb, K.; Thiele, L.; Laumanns, M.; Zitzler, E. (2001).
    Scalable Test Problems for Evolutionary Multi-Objective Optimization,
    Technical Report, Computer Engineering and Networks Laboratory (TIK),
    Swiss Federal Institute of Technology (ETH).
    https://dx.doi.org/10.3929/ethz-a-004284199

.. [Deb2002] Deb, K.; Thiele, L.; Laumanns, M.; Zitzler, E. (2002).
    Scalable multi-objective optimization test problems, Proceedings of
    the IEEE Congress on Evolutionary Computation, pp. 825-830

"""
import copy
import math
import random

from benchmark.base import BoundConstraintsChecker, Individual
from benchmark.multiobjective import MultiObjectiveTestProblem

__all__ = ["DTLZ", "DTLZBaseProblem", "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4",
           "DTLZ5", "DTLZ6", "DTLZ7"]

class DTLZBaseProblem(MultiObjectiveTestProblem):
    """The base class for DTLZ problems."""

    def __init__(self, objective_function, num_objectives, num_variables, phenome_preprocessor=None, **kwargs):
        assert num_variables > num_objectives
        self.num_variables = num_variables
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        self.is_deterministic = True
        self.do_maximize = False
        MultiObjectiveTestProblem.__init__(self,
                                           objective_function,
                                           num_objectives,
                                           phenome_preprocessor=preprocessor,
                                           **kwargs)


    def get_optimal_solutions(self, max_number):
        k = self.num_variables - self.num_objectives + 1
        n = self.num_variables
        solutions = []
        rand_generator = random.Random()
        rand_generator.seed(2)
        # generate full factorial sample of position parameters
        param_combis = [[]]
        for i in range(n - k):
            new_combis = []
            for combi in param_combis:
                combi_copy = copy.deepcopy(combi)
                combi.append(1.0)
                combi_copy.append(0.0)
                new_combis.append(combi_copy)
            param_combis.extend(new_combis)
        for combi in param_combis:
            combi.extend([0.5] * k)
            solutions.append(Individual(combi))
        # fill up with random optimal solutions
        while len(solutions) < max_number:
            phenome = [0.0] * n
            for i in range(n - k):
                phenome[i] += rand_generator.random()
            for i in range(n - k, n):
                phenome[i] += 0.5
            solutions.append(Individual(phenome))
        # assert length is not exceeded
        return solutions[0:max_number]



class DTLZ1(DTLZBaseProblem):
    """The DTLZ1 problem."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        DTLZBaseProblem.__init__(self,
                                 self.objective_function,
                                 num_objectives,
                                 num_variables,
                                 **kwargs)


    def g(self, phenome):
        """The g function of DTLZ1."""
        g = 0.0
        n = self.num_variables
        k = n - self.num_objectives + 1
        twenty_pi = 20.0 * math.pi
        for i in range(n - k + 1, n + 1):
            x = phenome[i-1]
            g += pow(x - 0.5, 2) - math.cos(twenty_pi * (x - 0.5))
        g = 100.0 * (k + g)
        return g


    def objective_function(self, phenome):
        num_variables = self.num_variables
        assert len(phenome) == num_variables
        num_objectives = self.num_objectives
        g = self.g(phenome)
        obj_values = []
        for i in range(1, num_objectives + 1):
            f = 0.5 * (1.0 + g)
            for j in range(1, num_objectives - i + 1):
                f *= phenome[j-1]
            if i > 1:
                f *= 1.0 - phenome[(num_variables - i + 1) - 1]
            obj_values.append(f)
        return obj_values


    def get_optimal_solutions(self, max_number):
        raise NotImplementedError("Optimal solutions are unknown.")



class DTLZ234_f:
    """The objective function of DTLZ2, 3, and 4."""

    def __init__(self, num_objectives, g, alpha):
        self.num_objectives = num_objectives
        self.g = g
        self.alpha = alpha


    def __call__(self, phenome):
        """Evaluate `phenome`."""
        num_objectives = self.num_objectives
        g_value = self.g(phenome)
        half_pi = math.pi / 2.0
        alpha = self.alpha
        obj_values = []
        for i in range(1, num_objectives + 1):
            f = 1.0 + g_value
            for j in range(1, num_objectives - i + 1):
                f *= math.cos(pow(phenome[j-1], alpha) * half_pi)
            if i > 1:
                index = (num_objectives - i + 1) - 1
                f *= math.sin(pow(phenome[index], alpha) * half_pi)
            obj_values.append(f)
        return obj_values



class DTLZ3(DTLZBaseProblem):
    """The DTLZ3 problem."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        DTLZBaseProblem.__init__(self,
                                 DTLZ234_f(num_objectives, self.g, 1),
                                 num_objectives,
                                 num_variables,
                                 **kwargs)


    def g(self, phenome):
        """The g function of DTLZ3."""
        g = 0.0
        n = self.num_variables
        k = n - self.num_objectives + 1
        twenty_pi = 20.0 * math.pi
        for i in range(n - k + 1, n + 1):
            x = phenome[i-1]
            g += pow(x - 0.5, 2) - math.cos(twenty_pi * (x - 0.5))
        g = 100.0 * (k + g)
        return g



class DTLZ4(DTLZBaseProblem):
    """The DTLZ4 problem."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        DTLZBaseProblem.__init__(self,
                                 DTLZ234_f(num_objectives, self.g, 100),
                                 num_objectives,
                                 num_variables,
                                 **kwargs)


    def g(self, phenome):
        """The g function of DTLZ4."""
        g = 0.0
        n = self.num_variables
        k = n - self.num_objectives + 1
        for i in range(n - k + 1, n + 1):
            g += math.pow(phenome[i-1] - 0.5, 2)
        return g



class DTLZ2(DTLZ4):
    """DTLZ2 is a special case of DTLZ4 with alpha = 1."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        DTLZ4.__init__(self,
                       num_objectives,
                       num_variables,
                       **kwargs)
        self.objective_function = DTLZ234_f(num_objectives, self.g, 1)



class DTLZ5(DTLZBaseProblem):
    """The DTLZ5 problem."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        DTLZBaseProblem.__init__(self,
                                 self.objective_function,
                                 num_objectives,
                                 num_variables,
                                 **kwargs)


    def g(self, phenome):
        """The g function of DTLZ5."""
        g = 0.0
        n = self.num_variables
        k = n - self.num_objectives + 1
        for i in range(n - k + 1, n + 1):
            g += math.pow(phenome[i-1] - 0.5, 2)
        return g


    def compute_theta(self, phenome, g_value):
        """Transform decision variables."""
        theta = [0.0] * len(phenome)
        t = math.pi / (4.0 * (1.0 + g_value))
        theta[0] = phenome[0] * math.pi / 2.0
        for i in range(2, self.num_objectives):
            theta[i-1] = t * (1.0 + 2.0 * g_value * phenome[i-1])
        return theta


    def objective_function(self, phenome):
        num_variables = self.num_variables
        assert len(phenome) == num_variables
        num_objectives = self.num_objectives
        g_value = self.g(phenome)
        theta = self.compute_theta(phenome, g_value)
        obj_values = []
        for i in range(1, num_objectives + 1):
            f = 1.0 + g_value
            for j in range(1, num_objectives - i + 1):
                f *= math.cos(theta[j-1])
            if i > 1:
                f *= math.sin(theta[(num_objectives - i + 1) - 1])
            obj_values.append(f)
        return obj_values


    def get_optimal_solutions(self, max_number):
        k = self.num_variables - self.num_objectives + 1
        n = self.num_variables
        solutions = []
        rand_generator = random.Random()
        rand_generator.seed(2)
        # generate full factorial sample of position parameters
        param_combis = [[]]
        for i in range(1):
            new_combis = []
            for combi in param_combis:
                combi_copy = copy.deepcopy(combi)
                combi.append(1.0)
                combi_copy.append(0.0)
                new_combis.append(combi_copy)
            param_combis.extend(new_combis)
        for combi in param_combis:
            combi.extend([0.5] * (n - 1))
            solutions.append(Individual(combi))
        # fill up with random optimal solutions
        while len(solutions) < max_number:
            phenome = [0.0] * n
            phenome[1] += rand_generator.random()
            for i in range(1, n):
                phenome[i] += 0.5
            solutions.append(Individual(phenome))
        # assert length is not exceeded
        return solutions[0:max_number]



class DTLZ6(DTLZ5):
    """DTLZ6 only differs from DTLZ5 in the g function."""

    def g(self, phenome):
        """The g function of DTLZ6."""
        g = 0.0
        n = self.num_variables
        k = n - self.num_objectives + 1
        for i in range(n - k + 1, n + 1):
            g += pow(phenome[i-1], 0.1)
        return g



class DTLZ7(DTLZBaseProblem):
    """The DTLZ7 problem."""

    def __init__(self, num_objectives, num_variables, **kwargs):
        DTLZBaseProblem.__init__(self,
                                 self.objective_function,
                                 num_objectives,
                                 num_variables,
                                 **kwargs)


    def g(self, phenome):
        """The g function of DTLZ7."""
        g = 0.0
        n = self.num_variables
        k = n - self.num_objectives + 1
        for i in range(n - k + 1, n + 1):
            g += phenome[i-1]
        return 1.0 + 9.0 * g / k


    def objective_function(self, phenome):
        num_variables = self.num_variables
        assert len(phenome) == num_variables
        num_objectives = self.num_objectives
        g_value = self.g(phenome)
        obj_values = [phenome[i] for i in range(num_objectives)]
        h = 0
        for j in range(1, num_objectives):
            h += obj_values[j-1] / (1.0 + g_value) * (1.0 + math.sin(3.0 * math.pi * obj_values[j-1]))
        h = num_objectives - h
        obj_values[num_objectives - 1] = (1.0 + g_value) * h
        return obj_values


    def get_optimal_solutions(self, max_number):
        raise NotImplementedError("Optimal solutions are unknown.")



class DTLZ(list):
    """The test problem collection DTLZ.

     This class inherits from :class:`list` and fills itself with seven
     DTLZ problems with the chosen configuration. The arguments to the
     constructor are passed through to the problem classes.

    """
    def __init__(self, num_objectives, num_variables, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_objectives : int
            The number of objectives for the optimization problems.
        num_variables : int
            The number of decision variables of the problems.
        kwargs
            Arbitrary keyword arguments, passed through to the constructors
            of the single DTLZ problems.

        """
        dtlz1 = DTLZ1(num_objectives, num_variables, **kwargs)
        dtlz2 = DTLZ2(num_objectives, num_variables, **kwargs)
        dtlz3 = DTLZ3(num_objectives, num_variables, **kwargs)
        dtlz4 = DTLZ4(num_objectives, num_variables, **kwargs)
        dtlz5 = DTLZ5(num_objectives, num_variables, **kwargs)
        dtlz6 = DTLZ6(num_objectives, num_variables, **kwargs)
        dtlz7 = DTLZ7(num_objectives, num_variables, **kwargs)
        list.__init__(self, [dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7])
