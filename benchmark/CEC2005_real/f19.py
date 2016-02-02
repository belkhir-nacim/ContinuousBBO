
from benchmark.CEC2005_real.f18 import F18



class F19(F18):
    """F18, but with a narrow basin for the global optimum."""

    sigmas = [0.1, 2.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]

    lambdas = [0.1 * 5.0 / 32.0, 5.0 / 32.0, 2.0 * 1, 1.0, 2.0 * 5.0 / 100.0,
               5.0 / 100.0, 2.0 * 10.0, 10.0, 2.0 * 5.0 / 60.0, 5.0 / 60.0]




