"""
Common binary test problems.
"""

from benchmark.base import TestProblem, Individual



def one_max(phenome):
    """The bare-bones one-max function."""
    return sum(phenome)



def leading_ones(phenome):
    """The bare-bones leading-ones function."""
    ret = 0
    i = 0
    while i < len(phenome) and phenome[i] == 1:
        ret += 1
        i += 1
    return ret



def trailing_zeros(phenome):
    """The bare-bones trailing-zeros function."""
    ret = 0
    i = len(phenome) - 1
    while i >= 0 and phenome[i] == 0:
        ret += 1
        i -= 1
    return ret



class BinaryChecker:
    """A pre-processor for checking if a phenome is binary.

    .. note:: This class makes use of the decorator design pattern for
        potential chaining of pre-processors, see
        https://en.wikipedia.org/wiki/Decorator_pattern

    """
    def __init__(self, num_variables=None, previous_preprocessor=None):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            Optionally require a specific number of phenes.
        previous_preprocessor : callable, optional
            Another callable that processes the phenome before this one
            does.

        """
        self.num_variables = num_variables
        self.previous_preprocessor = previous_preprocessor


    def __call__(self, phenome):
        """Check constraints and raise exception if necessary."""
        if self.previous_preprocessor is not None:
            phenome = self.previous_preprocessor(phenome)
        if self.num_variables is not None:
            assert len(phenome) == self.num_variables
        for phene in phenome:
            assert phene in (0, 1)
        return phenome



class OneMax(TestProblem):
    """The most simple binary optimization problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            The search space dimension.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome. Modifications should only be applied to a copy
            of the input. The (modified) phenome must be returned.
            Default behavior is to do no processing.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        preprocessor = BinaryChecker(num_variables, phenome_preprocessor)
        TestProblem.__init__(self,
                             one_max,
                             num_objectives=1,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.num_variables = num_variables
        self.is_deterministic = True
        self.do_maximize = True


    def get_optimal_solutions(self, max_number=None):
        """Return the optimal solution.

        .. note:: The returned solution does not yet contain the objective
            values.

        Returns
        -------
        solutions : list of Individual

        """
        assert max_number is None or max_number > 0
        opt = Individual([1] * self.num_variables)
        return [opt]


    get_locally_optimal_solutions = get_optimal_solutions



class LeadingOnes(TestProblem):
    """Counts the number of contiguous ones from the start of the bit-string."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            The search space dimension.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome. Modifications should only be applied to a copy
            of the input. The (modified) phenome must be returned.
            Default behavior is to do no processing.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        preprocessor = BinaryChecker(num_variables, phenome_preprocessor)
        TestProblem.__init__(self,
                             leading_ones,
                             num_objectives=1,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.num_variables = num_variables
        self.is_deterministic = True
        self.do_maximize = True


    def get_optimal_solutions(self, max_number=None):
        """Return the optimal solution.

        .. note:: The returned solution does not yet contain the objective
            values.

        Returns
        -------
        solutions : list of Individual

        """
        assert max_number is None or max_number > 0
        opt = Individual([1] * self.num_variables)
        return [opt]


    get_locally_optimal_solutions = get_optimal_solutions



class LeadingOnesTrailingZeros(TestProblem):
    """A bi-objective binary problem."""

    def __init__(self, num_variables=30, phenome_preprocessor=None, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            The search space dimension.
        phenome_preprocessor : callable, optional
            A callable potentially applying transformations or checks to
            the phenome. Modifications should only be applied to a copy
            of the input. The (modified) phenome must be returned.
            Default behavior is to do no processing.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        preprocessor = BinaryChecker(num_variables, phenome_preprocessor)
        TestProblem.__init__(self,
                             [leading_ones, trailing_zeros],
                             num_objectives=2,
                             phenome_preprocessor=preprocessor,
                             **kwargs)
        self.num_variables = num_variables
        self.is_deterministic = True
        self.do_maximize = True


    def get_optimal_solutions(self, max_number=None):
        """Return Pareto-optimal solutions.

        .. note:: The returned solutions do not yet contain the objective
            values.

        Parameters
        ----------
        max_number : int, optional
            Optionally restrict the number of solutions.

        Returns
        -------
        solutions : list of Individual
            The Pareto-optimal solutions

        """
        assert max_number is None or max_number > 0
        individuals = []
        for i in range(self.num_variables + 1):
            opt = Individual([1] * i + [0] * (self.num_variables - i))
            individuals.append(opt)
        if max_number is not None:
            individuals = individuals[:max_number]
        return individuals
