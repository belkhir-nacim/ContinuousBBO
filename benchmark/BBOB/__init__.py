"""

A set of 24 noiseless continuous optimization  function is used for the BBOB testbench

References
----------
Auger, A., Finck, S., Hansen, N., & Ros, R. (2010). BBOB 2009: Comparison tables of all algorithms on all noiseless functions.
Hansen, N., Auger, A., Finck, S., & Ros, R. (2010). Real-parameter black-box optimization benchmarking 2010: Experimental setup.
"""

from benchmark.BBOB import bbobbenchmarks as bn
from benchmark.base import TestProblem, BoundConstraintsChecker,Individual



class Function_BBOB(TestProblem):


    def __init__(self, num_variables,fid,iid=1, phenome_preprocessor=None, **kwargs):
        self.is_deterministic = True
        self.do_maximize = False
        self.num_variables = num_variables
        self.min_bounds = [-5] * num_variables
        self.max_bounds = [5] * num_variables
        bounds = (self.min_bounds, self.max_bounds)
        self.fitness, self.best = bn.instantiate(fid,iid)
        preprocessor = BoundConstraintsChecker(bounds, phenome_preprocessor)
        TestProblem.__init__(self, self.objective_function,
                             phenome_preprocessor=preprocessor,
                             **kwargs)


    def objective_function(self, phenome):
        assert len(phenome) == self.num_variables
        obj_value = self.fitness(phenome)
        return obj_value

    def get_optimal_solutions(self, max_number=None):
        return self.best



def bbob_bound(fid):
    return (-5,5)