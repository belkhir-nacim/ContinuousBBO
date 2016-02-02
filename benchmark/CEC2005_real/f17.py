
import random

from benchmark.CEC2005_real.f16 import F16



class F17(F16):
    """Rotated hybrid composition function F16 with noise."""

    def __init__(self, num_variables, **kwargs):
        F16.__init__(self, num_variables, **kwargs)
        self.is_deterministic = False


    def objective_function(self, phenome):
        obj_value = F16.objective_function(self, phenome) - self.bias
        obj_value *= (1.0 + 0.2 * random.gauss(0.0, 1.0))
        return obj_value + self.bias



