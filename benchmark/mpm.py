"""
This module contains the second version of the multiple-peaks model (MPM).
This test problem can be parametrized in many different ways. It is also
able to enumerate all local optima and to (approximately) determine which
attraction basin a point is located in. These features make it especially
well suited for research in multilocal optimization.
"""

import copy
import math
import random

import numpy as np

from benchmark.base import TestProblem, Individual, reflect

__all__ = ["MultiplePeaksModel2"]

class Peak(list):
    """Helper class maintaining the data structures needed for one peak."""

    def __init__(self,
                 num_variables,
                 height,
                 shape,
                 radius,
                 position=None,
                 min_bounds=None,
                 max_bounds=None,
                 rotated=True):
        """Constructor.

        Parameters
        ----------
        num_variables : int
            The search space dimension.
        height : float
            The height of this peak.
        shape : float
            Determines how pointed the peak is. A value of 2 means a
            quadratic peak.
        radius : float
            Controls the extent in the search space.
        position : array_like, optional
            A point in the search space that will attain the minimum of
            this peak. If omitted, a position will be drawn random
            uniformly in the search space.
        min_bounds : array_like, optional
            Lower bounds of the search space. Default is the zero vector.
        max_bounds : array_like, optional
            Upper bounds of the search space. Default is the one vector.
        rotated : bool, optional
            If a random covariance matrix shall be generated. Default is
            True.

        """
        assert num_variables > 0
        assert 0 <= height and height <= 1
        assert shape > 0
        assert radius > 0
        # shortcuts and initialization
        cos = math.cos
        sin = math.sin
        runi = random.uniform
        if position is None:
            position = np.random.rand(num_variables).tolist()
        if min_bounds is None:
            min_bounds = [0.0] * num_variables
        if max_bounds is None:
            max_bounds = [1.0] * num_variables
        # generate random rotation matrix
        rotation_matrix = np.eye(num_variables)
        if rotated:
            quarter_pi = math.pi / 4.0
            for j in range(num_variables - 1):
                for k in range(j + 1, num_variables):
                    temp = np.eye(num_variables)
                    alpha = runi(-quarter_pi, quarter_pi)
                    temp[j, j] = cos(alpha)
                    temp[j, k] = sin(alpha)
                    temp[k, j] = -sin(alpha)
                    temp[k, k] = cos(alpha)
                    rotation_matrix = np.dot(rotation_matrix, temp)
        # generate inverse 'covariance' matrix from rotation matrix
        variance_range = (np.array(max_bounds) - np.array(min_bounds)) / 20.0
        scaled_diag_values = np.random.rand(num_variables) * variance_range + variance_range * 0.05
        self.D = np.dot(np.dot(rotation_matrix.T, np.diag(scaled_diag_values)), rotation_matrix)
        self.D = np.linalg.inv(self.D)
        # other data
        list.__init__(self, position)
        self.height = height
        self.shape = shape
        self.radius = radius



class MultiplePeaksModel2(TestProblem):
    """A test problem with a controllable number of local optima.

    The mathematical definition can be found in [Wessing2015]_.

    References
    ----------
    .. [Wessing2015] S. Wessing (2015). The Multiple Peaks Model 2.
        Algorithm Engineering Report TR15-2-001, Technische Universitaet
        Dortmund.
        https://ls11-www.cs.tu-dortmund.de/_media/techreports/tr15-01.pdf
    """

    def __init__(self, num_variables=10, peaks=None, **kwargs):
        """Constructor.

        Parameters
        ----------
        num_variables : int, optional
            The search space dimension.
        peaks : sequence of Peak
            Previously prepared peaks. If None, a few peaks are generated
            randomly.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor
            of the super class.

        """
        self.min_bounds = [0.0] * num_variables
        self.max_bounds = [1.0] * num_variables
        TestProblem.__init__(self,
                             self.objective_function,
                             num_objectives=1,
                             **kwargs)
        self.num_variables = num_variables
        self.peaks = peaks
        if peaks is None:
            self.peaks = self.rand_uniform_peaks(num_variables=num_variables)
        self.is_deterministic = True


    @classmethod
    def create_instance_with_exact_num_optima(cls,
                                              num_optima,
                                              num_variables,
                                              topology,
                                              shape_height_correlation,
                                              height_range=None,
                                              shape_range=None,
                                              radius_range=None,
                                              rotated_peaks=True,
                                              **kwargs):
        """Factory method for creating a problem instance.

        This method sequentially adds peaks to the set, until a given
        number of local optima is reached. In the worst case, this will
        lead to a cubic run time in the number of optima.

        Parameters
        ----------
        num_optima : int
            The number of desired local optima.
        num_variables : int
            The number of decision variables of the search space.
        topology : str
            Must be 'random' or 'funnel'. If 'funnel', the peak heights
            are reordered to decrease together with the distance from
            the best peak. The original peaks are not modified.
        shape_height_correlation : int
            Must be -1, 0, or 1. If this value is not zero, also the shape
            parameters are reordered to be perfectly (anti)correlated with
            the peak height.
        height_range : tuple of float, optional
            For peaks that are not global optima, their function value is
            drawn random uniformly from this range.
        shape_range : tuple of float, optional
            The Mahalanobis distance is exponentiated with a value drawn
            random uniformly from this range.
        radius_range : tuple of float, optional
            The exponentiated Mahalanobis distance is divided by a value
            drawn random uniformly from this range.
        rotated : bool, optional
            If true, a random rotation is applied to each peak.

        Returns
        -------
        problem : MultiplePeaksModel2 instance

        """
        if height_range is None:
            height_range = (0.5, 0.99)
        if shape_range is None:
            shape_range = (1.5, 2.5)
        if radius_range is None:
            radius_range = (0.25 * math.sqrt(num_variables), 0.5 * math.sqrt(num_variables))
        global_opt = cls.rand_uniform_peaks(1,
                                            num_variables,
                                            num_global_opt=1,
                                            height_range=height_range,
                                            shape_range=shape_range,
                                            radius_range=radius_range,
                                            rotated=rotated_peaks)[0]
        if topology == "random":
            peaks = cls.rand_uniform_peaks(num_optima - 1,
                                           num_variables,
                                           num_global_opt=0,
                                           height_range=height_range,
                                           shape_range=shape_range,
                                           radius_range=radius_range,
                                           rotated=rotated_peaks)
        elif topology == "funnel":
            peaks = cls.clustered_peaks(num_optima - 1,
                                        num_variables,
                                        num_global_opt=0,
                                        height_range=height_range,
                                        shape_range=shape_range,
                                        radius_range=radius_range,
                                        rotated=rotated_peaks,
                                        cluster_center=global_opt)
        else:
            raise ValueError("unknown problem topology " + str(topology))
        peaks.append(global_opt)
        problem = cls.create_instance(peaks, topology, shape_height_correlation, **kwargs)
        current_num_optima = len(problem.get_locally_optimal_solutions())
        factor = 1.0
        while current_num_optima < num_optima * 0.8 and factor > 0.01:
            peaks_copy = copy.deepcopy(peaks)
            for peak in peaks_copy:
                peak.radius *= 0.95
            problem = cls.create_instance(peaks, topology, shape_height_correlation, **kwargs)
            current_num_optima = len(problem.get_locally_optimal_solutions())
            if current_num_optima < num_optima:
                peaks = peaks_copy
            factor *= 0.95
        radius_range = (radius_range[0] * factor, radius_range[1] * factor)
        problem = cls.create_instance(peaks, topology, shape_height_correlation, **kwargs)
        current_num_optima = len(problem.get_locally_optimal_solutions())
        prev_num_optima = current_num_optima
        while current_num_optima < num_optima:
            while current_num_optima != prev_num_optima + 1:
                # generate list of 1 random new peak
                if topology == "random":
                    new_peaks = cls.rand_uniform_peaks(1,
                                                       num_variables,
                                                       num_global_opt=0,
                                                       height_range=height_range,
                                                       shape_range=shape_range,
                                                       radius_range=radius_range,
                                                       rotated=rotated_peaks)
                elif topology == "funnel":
                    new_peaks = cls.clustered_peaks(1,
                                                    num_variables,
                                                    num_global_opt=0,
                                                    height_range=height_range,
                                                    shape_range=shape_range,
                                                    radius_range=radius_range,
                                                    rotated=rotated_peaks,
                                                    cluster_center=global_opt)
                problem = cls.create_instance(peaks + new_peaks, topology, shape_height_correlation, **kwargs)
                current_num_optima = len(problem.get_locally_optimal_solutions())
            prev_num_optima = current_num_optima
            peaks.extend(new_peaks)
        return problem


    @classmethod
    def create_instance(cls, peaks, topology="random", shape_height_correlation=0, **kwargs):
        """Factory method for creating a problem instance.

        Parameters
        ----------
        peaks : sequence of Peak
            Previously prepared peaks.
        topology : str, optional
            Must be 'random' or 'funnel'. If 'funnel', the peak heights
            are reordered to decrease together with the distance from
            the best peak. The original peaks are not modified.
        shape_height_correlation : int, optional
            Must be -1, 0, or 1. If this value is not zero, also the shape
            parameters are reordered to be perfectly (anti)correlated with
            the peak height.
        kwargs
            Arbitrary keyword arguments, passed through to the constructor.

        Returns
        -------
        problem : MultiplePeaksModel2 instance

        """
        assert shape_height_correlation in (-1, 0, 1)
        # attention: deepcopy is important because peak objects are
        # modified in the following
        peaks_copy = copy.deepcopy(peaks)
        num_variables = len(peaks_copy[0])
        for peak in peaks_copy:
            assert len(peak) == num_variables
        if topology == "funnel":
            # sort peaks according to height (descending)
            peaks_decorated = [(peak.height, peak) for peak in peaks_copy]
            peaks_decorated.sort(reverse=True)
            heights = [height for height, _ in peaks_decorated]
            # set global optimum as center
            center = peaks_decorated[0][1]
            peaks_decorated = []
            for peak in peaks_copy:
                dist = sum((c - p) ** 2 for c, p in zip(center, peak))
                peaks_decorated.append((dist, peak))
            peaks_decorated.sort()
            peaks_copy = [peak for _, peak in peaks_decorated]
            # make heights anti-correlated to distance to center
            for height, peak in zip(heights, peaks_copy):
                peak.height = height
        elif topology == "random":
            # nothing to do, everything is random
            pass
        else:
            raise Exception("undefined topology")
        if shape_height_correlation != 0:
            shapes = [peak.shape for peak in peaks_copy]
            anti_correlated = (shape_height_correlation == -1)
            shapes.sort(reverse=anti_correlated)
            peaks_decorated = [(peak.height, peak) for peak in peaks_copy]
            peaks_decorated.sort()
            peaks_copy = [peak for _, peak in peaks_decorated]
            for peak, shape in zip(peaks_copy, shapes):
                peak.shape = shape
        problem = cls(num_variables, peaks_copy, **kwargs)
        return problem


    @classmethod
    def clustered_peaks(cls,
                        num_peaks=8,
                        num_variables=10,
                        num_global_opt=1,
                        height_range=(0.5, 0.99),
                        shape_range=(1.75, 2.25),
                        radius_range=(0.25, 0.5),
                        rotated=True,
                        cluster_center=None):
        """Create peaks with Gaussian distribution.

        Parameters
        ----------
        num_peaks : int, optional
            The number of peaks to generate.
        num_variables : int, optional
            The number of decision variables of the search space.
        num_global_opt : int, optional
            Determines how many peaks will be global optima.
        height_range : tuple of float, optional
            For peaks that are not global optima, their function value is
            drawn random uniformly from this range.
        shape_range : tuple of float, optional
            The Mahalanobis distance is exponentiated with a value drawn
            random uniformly from this range.
        radius_range : tuple of float, optional
            The exponentiated Mahalanobis distance is divided by a value
            drawn random uniformly from this range.
        rotated : bool, optional
            If true, a random rotation is applied to each peak.
        cluster_center : array_like, optional
            A vector that will be taken as expectation of the normal
            distribution. If omitted, a vector will be drawn random
            uniformly in the search space.

        Returns
        -------
        peaks : list of Peak

        """
        assert num_peaks >= 0
        assert num_variables > 0
        assert num_global_opt >= 0
        assert num_peaks >= num_global_opt
        if cluster_center is None:
            # determine random uniform cluster center
            cluster_center = np.random.rand(num_variables)
        peaks = []
        for _ in range(num_peaks):
            position = np.random.randn(num_variables) / 6.0
            position *= math.sqrt(num_variables)
            position += cluster_center
            # repair box constraint violations of position
            for j in range(num_variables):
                position[j] = reflect(position[j], 0.0, 1.0)
            # build peak
            peaks.append(Peak(num_variables,
                              random.uniform(*height_range),
                              random.uniform(*shape_range),
                              random.uniform(*radius_range),
                              position=position.tolist(),
                              rotated=rotated))
        global_optima = random.sample(peaks, num_global_opt)
        for opt in global_optima:
            opt.height = 1.0
        return peaks


    @classmethod
    def rand_uniform_peaks(cls,
                           num_peaks=8,
                           num_variables=10,
                           num_global_opt=1,
                           height_range=(0.5, 0.99),
                           shape_range=(1.75, 2.25),
                           radius_range=(0.25, 0.5),
                           rotated=True):
        """Create peaks with random uniform distribution.

        Parameters
        ----------
        num_peaks : int, optional
            The number of peaks to generate.
        num_variables : int, optional
            The number of decision variables of the search space.
        num_global_opt : int, optional
            Determines how many peaks will be global optima.
        height_range : tuple of float, optional
            For peaks that are not global optima, their function value is
            drawn random uniformly from this range.
        shape_range : tuple of float, optional
            The Mahalanobis distance is exponentiated with a value drawn
            random uniformly from this range.
        radius_range : tuple of float, optional
            The exponentiated Mahalanobis distance is divided by a value
            drawn random uniformly from this range.
        rotated : bool, optional
            If true, a random rotation is applied to each peak.

        Returns
        -------
        peaks : list of Peak

        """
        assert num_peaks >= 0
        assert num_variables > 0
        assert num_global_opt >= 0
        assert num_peaks >= num_global_opt
        num_remaining_peaks = num_peaks - num_global_opt
        runi = random.uniform
        peaks = []
        for _ in range(num_global_opt):
            peaks.append(Peak(num_variables,
                              1.0,
                              runi(*shape_range),
                              runi(*radius_range),
                              rotated=rotated))
        for _ in range(num_remaining_peaks):
            peaks.append(Peak(num_variables,
                              runi(*height_range),
                              runi(*shape_range),
                              runi(*radius_range),
                              rotated=rotated))
        return peaks


    @staticmethod
    def dist(phenome, peak):
        """Mahalanobis distance"""
        diff_vector = np.array(peak)
        diff_vector -= phenome
        return math.sqrt(np.dot(np.dot(diff_vector, peak.D), diff_vector))


    def g(self, phenome, peak):
        """Helper method for parametrizing a peak function."""
        dist = self.dist(phenome, peak)
        return peak.height / (1.0 + math.pow(dist, peak.shape) / peak.radius)


    def objective_function(self, phenome):
        """Aggregate the function values of all peak functions.

        Parameters
        ----------
        phenome : sequence of float
            The solution to be evaluated.

        Returns
        -------
        objective_value : float

        """
        assert len(phenome) == self.num_variables
        g = self.g
        phenome = np.array(phenome)
        max_g_value = max(g(phenome, peak) for peak in self.peaks)
        return 1.0 - max_g_value


    def get_active_peak(self, phenome):
        """Return the peak modeling the function at `phenome`."""
        g = self.g
        max_objective_value = float("-inf")
        active_peak = None
        for peak in self.peaks:
            objective_value = g(phenome, peak)
            if objective_value > max_objective_value:
                active_peak = peak
                max_objective_value = objective_value
        return active_peak


    def get_basin(self, phenome):
        """Return the peak in whose attraction basin `phenome` is located.

        Generally, the attraction basin may consist of several overlayed
        peaks. This method only yields an approximation, i.e., the returned
        peak may be not the one an ideal steepest descent algorithm would
        converge to.

        """
        get_active_peak = self.get_active_peak
        previous_peak = list(phenome)
        current_peak = get_active_peak(previous_peak)
        while previous_peak != current_peak:
            previous_peak = current_peak
            current_peak = get_active_peak(current_peak)
        return current_peak


    def get_peaks_sorted_by_importance(self):
        """Return all peaks together with importance information.

        In this implementation, the importance of a peak is defined as the
        difference between a peak's height and the best value of other peak
        functions at this location.

        Returns
        -------
        decorated_peaks : list of tuple
            A list of tuples (importance, peak) in descending order from
            high to low importance. Negative values indicate that the peak
            is not a local optimum.

        """
        g = self.g
        decorated_peaks = []
        for peak in self.peaks:
            this_g_value = g(peak, peak)
            other_g_values = [g(peak, other) for other in self.peaks if other is not peak]
            importance = this_g_value - max(other_g_values)
            decorated_peaks.append((importance, peak))
        decorated_peaks.sort(reverse=True)
        return decorated_peaks


    def get_locally_optimal_solutions(self, max_number=None):
        """Return locally optimal solutions (includes global ones).

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        get_active_peak = self.get_active_peak
        # test peaks
        local_optima = []
        peaks = np.vstack(self.peaks)
        for i, peak in enumerate(peaks):
            if get_active_peak(peak) is self.peaks[i]:
                local_optima.append(Individual(phenome=list(peak)))
        if max_number is not None:
            local_optima = local_optima[:max_number]
        return local_optima


    def get_optimal_solutions(self, max_number=None):
        """Return globally optimal solutions.

        Parameters
        ----------
        max_number : int, optional
            Potentially restrict the number of optima.

        Returns
        -------
        optima : list of Individual

        """
        # test peaks
        optima = []
        max_height = -1.0
        opt_phenomes = []
        for peak in self.peaks:
            if peak.height > max_height:
                opt_phenomes = [peak]
                max_height = peak.height
            elif peak.height == max_height:
                opt_phenomes.append(peak)
        for phenome in opt_phenomes:
            optima.append(Individual(phenome=list(phenome)))
        if max_number is not None:
            optima = optima[:max_number]
        return optima
