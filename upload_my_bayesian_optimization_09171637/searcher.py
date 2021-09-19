# coding=utf-8
import copy
import math
import random
import statistics
import warnings
import threading

import numpy as np
from sklearn.ensemble.base import _partition_estimators

from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble.forest import *
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize
from scipy.stats import norm

# Need to import the searcher abstract class, the following are essential
from sklearn.utils.validation import check_is_fitted

from thpo.abstract_searcher import AbstractSearcher


# class myRandomForestRegressor(RandomForestRegressor):
#     def __init__(self,
#                  n_estimators='warn',
#                  criterion="mse",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  min_impurity_split=None,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=None,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False):
#         super(RandomForestRegressor, self).__init__(
#             base_estimator=DecisionTreeRegressor(),
#             n_estimators=n_estimators,
#             estimator_params=("criterion", "max_depth", "min_samples_split",
#                               "min_samples_leaf", "min_weight_fraction_leaf",
#                               "max_features", "max_leaf_nodes",
#                               "min_impurity_decrease", "min_impurity_split",
#                               "random_state"),
#             bootstrap=bootstrap,
#             oob_score=oob_score,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             verbose=verbose,
#             warm_start=warm_start)
#
#         self.criterion = criterion
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.min_weight_fraction_leaf = min_weight_fraction_leaf
#         self.max_features = max_features
#         self.max_leaf_nodes = max_leaf_nodes
#         self.min_impurity_decrease = min_impurity_decrease
#         self.min_impurity_split = min_impurity_split
#     def predict(self, X):
#         """Predict regression target for X.
#
#         The predicted regression target of an input sample is computed as the
#         mean predicted regression targets of the trees in the forest.
#
#         Parameters
#         ----------
#         X : array-like or sparse matrix of shape = [n_samples, n_features]
#             The input samples. Internally, its dtype will be converted to
#             ``dtype=np.float32``. If a sparse matrix is provided, it will be
#             converted into a sparse ``csr_matrix``.
#
#         Returns
#         -------
#         y : array of shape = [n_samples] or [n_samples, n_outputs]
#             The predicted values.
#         """
#         check_is_fitted(self, 'estimators_')
#         # Check data
#         X = self._validate_X_predict(X)
#
#         # Assign chunk of trees to jobs
#         n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
#
#         # avoid storing the output of every estimator by summing them here
#         if self.n_outputs_ > 1:
#             y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
#         else:
#             y_hat = np.zeros((X.shape[0]), dtype=np.float64)
#
#         # Parallel loop
#         lock = threading.Lock()
#         Parallel(n_jobs=n_jobs, verbose=self.verbose,
#                  **_joblib_parallel_args(require="sharedmem"))(
#             delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
#             for e in self.estimators_)
#
#         y_hat /= len(self.estimators_)
#
#         return y_hat


class UtilityFunction(object):
    """
    This class mainly implements the collection function
    """

    def __init__(self, kind, kappa, x_i):
        self.kappa = kappa
        self.x_i = x_i

        self._iters_counter = 0

        if kind not in ['gp_ucb', 'gp_ei', 'gp_poi', 'rfr']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x_x, model, y_max):
        if self.kind == 'rfr':
            return self._rfr(x_x, model, y_max, self.x_i)
        if self.kind == 'gp_ucb':
            return self._ucb(x_x, model, self.kappa)
        if self.kind == 'gp_ei':
            return self._ei(x_x, model, y_max, self.x_i)
        if self.kind == 'gp_poi':
            return self._poi(x_x, model, y_max, self.x_i)

    @staticmethod
    def _rfr(x_x, model, y_max, x_i):
        rets = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(model)):
                rets.append(model[i].predict(x_x))
            mean = np.mean(rets)
            std = np.std(rets)+0.01
        a_a = (mean - y_max - x_i)
        z_z = a_a / std
        return a_a * norm.cdf(z_z) + std * norm.pdf(z_z)

    @staticmethod
    def _ucb(x_x, model, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = model.predict(x_x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        a_a = (mean - y_max - x_i)
        z_z = a_a / std
        return a_a * norm.cdf(z_z) + std * norm.pdf(z_z)

    @staticmethod
    def _poi(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        z_z = (mean - y_max - x_i) / std
        return norm.cdf(z_z)


class Searcher(AbstractSearcher):

    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher

        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.

                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.

        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)
        gp = GaussianProcessRegressor(
            kernel=RationalQuadratic(),
            alpha=1e-4,
            normalize_y=True,
            random_state=np.random.RandomState(1),
        )
        self.rfrn = 20
        self.rfr = []
        for i in range(self.rfrn):
            rfr = RandomForestRegressor(n_estimators=3)
            self.rfr.append(rfr)
        self.gp = gp
        # print(parameters_config)

    def init_param_group(self, n_suggestions):
        """ Suggest n_suggestions parameters in random form

        Args:
            n_suggestions: number of parameters to suggest in every iteration

        Return:
            next_suggestions: n_suggestions Parameters in random form
        """
        next_suggestions = [{p_name: p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)]
                             for p_name, p_conf in self.parameters_config.items()} for _ in range(n_suggestions)]

        return next_suggestions

    def parse_suggestions_history(self, suggestions_history):
        """ Parse historical suggestions to the form of (x_datas, y_datas), to obtain GP training data

        Args:
            suggestions_history: suggestions history

        Return:
            x_datas: Parameters
            y_datas: Reward of Parameters
        """
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def train_gp(self, x_datas, y_datas):
        """ train gp

        Args:
            x_datas: Parameters
            y_datas: Reward of Parameters

        Return:
            gp: Gaussian process regression
        """
        self.gp.fit(x_datas, y_datas)

    def train_rfr(self, x_datas, y_datas):
        for i in range(self.rfrn):
            self.rfr[i].fit(x_datas, y_datas)

    def random_sample(self):
        """ Generate a random sample in the form of [value_0, value_1,... ]

        Return:
            sample: a random sample in the form of [value_0, value_1,... ]
        """
        sample = [p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)] for p_name, p_conf
                  in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        return sample

    def get_bounds(self):
        """ Get sorted parameter space

        Return:
            _bounds: The sorted parameter space
        """

        def _get_param_value(param):
            value = [param['double_min_value'], param['double_max_value']]
            return value

        _bounds = np.array(
            [_get_param_value(item[1]) for item in sorted(self.parameters_config.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        return _bounds

    def acq_max(self, f_acq, model, y_max, bounds, num_warmup, num_starting_points):
        """ Produces the best suggested parameters

        Args:
            f_acq: Acquisition function
            gp: GaussianProcessRegressor
            y_max: Best reward in suggestions history
            bounds: The parameter boundary of the acquisition function
            num_warmup: The number of samples randomly generated for the collection function
            num_starting_points: The number of random samples generated for scipy.minimize

        Return:
            Return the current optimal parameters
        """
        # Warm up with random points
        x_tries = np.array([self.random_sample() for _ in range(int(num_warmup))])
        ys = f_acq(x_tries, model=model, y_max=y_max)
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
        # Explore the parameter space more throughly
        x_seeds = np.array([self.random_sample() for _ in range(int(num_starting_points))])
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -f_acq(x.reshape(1, -1), model=model, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")
            # See if success
            if not res.success:
                continue
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
        return np.clip(x_max, bounds[:, 0], bounds[:, 1]), max_acq

    def parse_suggestions(self, suggestions):
        """ Parse the parameters result

        Args:
            suggestions: Parameters

        Return:
            suggestions: The parsed parameters
        """

        def get_param_value(p_name, value):
            p_coords = self.parameters_config[p_name]['coords']
            if value in p_coords:
                return value
            else:
                subtract = np.abs([p_coord - value for p_coord in p_coords])
                min_index = np.argmin(subtract, axis=0)
                return p_coords[min_index]

        p_names = [p_name for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        suggestions = [{p_names[index]: suggestion[index] for index in range(len(suggestion))}
                       for suggestion in suggestions]

        suggestions = [{p_name: get_param_value(p_name, value) for p_name, value in suggestion.items()}
                       for suggestion in suggestions]
        return suggestions

    def suggest(self, suggestions_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters.

        Args:
            suggestions_history: a list of historical suggestion parameters and rewards, in the form of
                    [[Parameter, Reward], [Parameter, Reward] ... ]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                        Reward: a float type value

                    The parameters and rewards of each iteration are placed in suggestions_history in the order of iteration.
                        len(suggestions_history) = n_suggestion * iteration(current number of iteration)

                    For example:
                        when iteration = 2, n_suggestion = 2, then
                        [[{'p1': 0, 'p2': 0, 'p3': 0}, -222.90621774147272],
                         [{'p1': 0, 'p2': 1, 'p3': 3}, -65.26678723205647],
                         [{'p1': 2, 'p2': 2, 'p3': 2}, 0.0],
                         [{'p1': 0, 'p2': 0, 'p3': 4}, -105.8151893979122]]

            n_suggestion: int, number of suggestions to return

        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}

                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        if (suggestions_history is None) or (len(suggestions_history) <= 0):
            next_suggestions = self.init_param_group(n_suggestions)
        else:
            x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
            self.train_gp(x_datas, y_datas)
            self.train_rfr(x_datas, y_datas)
            _bounds = self.get_bounds()
            suggestions = []
            suggestions_candidate = []
            for index in range(max(n_suggestions * 5, 10)):
                utility_function = UtilityFunction(kind='gp_ei', kappa=(index + 1) * 2.567, x_i=index+1)
                suggestion, max_acq = self.acq_max(
                    f_acq=utility_function.utility,
                    model=self.gp,
                    y_max=y_datas.max(),
                    bounds=_bounds,
                    num_warmup=1000,
                    num_starting_points=50
                )
                suggestions_candidate.append((suggestion, max_acq))
            suggestions_candidate.sort(key=lambda y: y[1], reverse=True)
            for i in range(n_suggestions):
                suggestions.append(suggestions_candidate[i][0])

            suggestions = np.array(suggestions)
            suggestions = self.parse_suggestions(suggestions)
            next_suggestions = suggestions

        return next_suggestions
