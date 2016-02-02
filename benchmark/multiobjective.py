"""
Multiobjective test problems.

So far, this module only contains a base class for multiobjective test
problems.

"""

import diversipy.subset

from benchmark.base import TestProblem



class MultiObjectiveTestProblem(TestProblem):
    """Base class for multiobjective test problems.

    This class only adds functionality to sample the Pareto-front
    relatively uniformly.

    """
    def sample_pareto_front(self, num_points, oversampling_factor=100):
        """Sample the whole Pareto-front.

        This method works by obtaining ``num_points * oversampling_factor``
        Pareto-optimal points in the search space from
        :func:`get_optimal_solutions<benchmark.base.TestProblem.get_optimal_solutions>`,
        evaluating them, and then selecting num_points of them with a
        uniform distribution in objective space by the algorithm in
        :func:`diversipy.subset.psa_partition`.

        Parameters
        ----------
        num_points : int
            The number of points to sample on the Pareto-front.
        oversampling_factor : int or float, optional
            A parameter controlling the uniformity of the points'
            distribution on the Pareto-front. The higher this value, the
            higher is the chance of obtaining a uniform distribution.

        Returns
        -------
        selected_solutions : list of Individual

        """
        # generate Pareto-optimal solutions, oversample the search space
        solutions = self.get_optimal_solutions(int(num_points * oversampling_factor))
        for solution in solutions:
            # evaluate without counting
            solution.objective_values = self.objective_function(solution.phenome)
        # add best solution for each objective
        selected_indices = set()
        if num_points > self.num_objectives:
            for i in range(self.num_objectives):
                min_objective = float("inf")
                min_index = None
                for j, solution in enumerate(solutions):
                    if solution.objective_values[i] < min_objective:
                        min_index = j
                        min_objective = solution.objective_values[i]
                selected_indices.add(min_index)
        # select remaining solutions uniformly
        available_indices = list(set(range(len(solutions))).difference(selected_indices))
        points = [solution.objective_values for solution in solutions]
        num_remaining = num_points - len(selected_indices)
        clusters = diversipy.subset.psa_partition(points, num_remaining, available_indices)
        selected_indices = list(selected_indices)
        selected_indices += [cluster.obtain_representative_index() for cluster in clusters]
        if self.num_objectives == 2:
            def sortkey(index):
                return points[index]
            selected_indices.sort(key=sortkey)
        selected_solutions = [solutions[i] for i in selected_indices]
        return selected_solutions

