
from benchmark.base import identity
from benchmark.CEC2005_real.f24 import F24



class F25(F24):
    """Rotated hybrid composition function F24 without bounds."""

    def __init__(self, num_variables, matrices=None, offsets=None, **kwargs):
        F24.__init__(self, num_variables, matrices, offsets, **kwargs)
        self.phenome_preprocessor = self.phenome_preprocessor.previous_preprocessor
        if self.phenome_preprocessor is None:
            self.phenome_preprocessor = identity
        self.min_bounds = None
        self.max_bounds = None
