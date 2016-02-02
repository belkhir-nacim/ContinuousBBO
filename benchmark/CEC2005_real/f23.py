
from benchmark.CEC2005_real.f21 import F21
from benchmark.CEC2005_real.helper import my_x_round


class F23(F21):
    """Non-Continuous rotated hybrid composition function F21."""

    def objective_function(self, phenome):
        phenome = [my_x_round(phene, offset) for phene, offset in zip(phenome, self.offsets[0])]
        assert len(phenome) == self.num_variables
        return self.hybrid_composition_function(phenome) + self.bias
