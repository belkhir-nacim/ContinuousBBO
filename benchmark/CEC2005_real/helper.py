
import math
import random

import numpy as np

from benchmark.real import sphere, schaffer6


class HybridCompositionFunction:

    def __init__(self, num_variables, basic_functions, matrices, sigmas, lambdas, biases, offsets, c, name=None):
        self.num_variables = num_variables
        self.basic_functions = basic_functions
        self.matrices = matrices
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.biases = biases
        self.offsets = offsets
        self.c = c
        # sanity checks
        for matrix in matrices:
            assert matrix.shape == (num_variables, num_variables)
        assert len(basic_functions) == len(matrices)
        assert len(matrices) == len(sigmas)
        assert len(sigmas) == len(lambdas)
        assert len(lambdas) == len(biases)
        assert len(biases) == len(offsets)
        # calculate/estimate the f_max for all the functions involved
        self.f_max = []
        for i, basic_function in enumerate(basic_functions):
            test_point = np.array([5.0 / lambdas[i] for j in range(num_variables)])
            test_point = np.dot(test_point, matrices[i])
            self.f_max.append(abs(basic_function(test_point)))


    def __call__(self, phenome):
        # shortcuts
        num_variables = self.num_variables
        sigmas = self.sigmas
        lambdas = self.lambdas
        basic_functions = self.basic_functions
        matrices = self.matrices
        f_max = self.f_max
        biases = self.biases
        # get the raw weights
        w_max = float("-inf")
        z = []
        weights = []
        for i in range(len(basic_functions)):
            z.append(np.array([phene - offset for offset, phene in zip(self.offsets[i], phenome)]))
            sum_of_squares = 0.0
            for j in range(num_variables):
                sum_of_squares += z[i][j] ** 2
            weight = math.exp(-1.0 * sum_of_squares / (2.0 * num_variables * sigmas[i] ** 2))
            weights.append(weight)
            if w_max < weights[i]:
                w_max = weights[i]
        # modify the weights
        w1m_max_pow = 1.0 - math.pow(w_max, 10.0)
        for i in range(len(basic_functions)):
            if weights[i] != w_max:
                weights[i] *= w1m_max_pow
        # normalize the weights
        weight_sum = sum(weights)
        for i in range(len(basic_functions)):
            weights[i] /= weight_sum
        # calculate objective value
        sum_f = 0.0
        for i, basic_function in enumerate(basic_functions):
            for j in range(num_variables):
                z[i][j] /= lambdas[i]
            z[i] = np.dot(z[i], matrices[i])
            sum_f += weights[i] * (self.c * basic_function(z[i]) / f_max[i] + biases[i])
        return sum_f



def my_round(x):
    """Ensure old-fashioned rounding behavior."""
    return math.copysign(int(abs(x) + 0.5), x)



def my_x_round(x, offset=0.0):
    if abs(x - offset) < 0.5:
        return x
    else:
        return my_round(2.0 * x) / 2.0



def generalized_schaffer6(phenome):
    """A generalization of Schaffer's function 6 for arbitrary dimensions."""
    result = sum(schaffer6(phenome[i-1:i+1]) for i in range(1, len(phenome)))
    result += schaffer6([phenome[-1], phenome[0]])
    return result



def noisy_sphere(phenome):
    return sphere(phenome) * (1.0 + 0.1 * abs(random.gauss(0.0, 1.0)))


