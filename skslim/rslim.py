import logging
import os
from contextlib import redirect_stdout
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit
from riskslim import CoefficientSet, run_lattice_cpa
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

logger = logging.getLogger("default")


class RiskSlim(BaseEstimator, ClassifierMixin):
    def __init__(self, max_score=3, min_score=None, C=1e-3, random_state=None, timeout=900):
        self.max_score = max_score
        self.min_score = min_score
        self.C = C
        self.random_state = random_state
        self.timeout = timeout

        self.computed_min_score = -max_score if min_score is None else min_score
        self.model = None  # type: Optional[np.array]
        self.solution_status_code = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        # add bias feature
        X = np.insert(arr=X, obj=0, values=np.ones(X.shape[0]), axis=1)
        # spread target to -1,1
        y[y == 0] = -1

        # create coefficient set and set the value of the offset parameter
        feature_names = ["(Intercept)"] + [str(i) for i in range(X.shape[1] - 1)]
        coef_set = CoefficientSet(variable_names=feature_names, ub=self.max_score, lb=self.computed_min_score, sign=0,
                                  print_flag=False)
        coef_set.update_intercept_bounds(X=X, y=y, max_offset=50)

        constraints = {
            'L0_min': 0,
            'L0_max': 5,
            'coef_set': coef_set,
        }

        # major settings (see riskslim_ex_02_complete for full set of options)
        settings = {
            # Problem Parameters
            'c0_value': self.C,
            #
            # LCPA Settings
            'max_runtime': self.timeout,  # max runtime for LCPA
            'max_tolerance': np.finfo('float').eps,
            # tolerance to stop LCPA (set to 0 to return provably optimal solution)
            'display_cplex_progress': False,  # print CPLEX progress on screen
            'loss_computation': 'fast',  # how to compute the loss function ('normal','fast','lookup')
            #
            # LCPA Improvements
            'round_flag': True,  # round continuous solutions with SeqRd
            'polish_flag': True,  # polish integer feasible solutions with DCD
            'chained_updates_flag': True,  # use chained updates
            'add_cuts_at_heuristic_solutions': True,
            # add cuts at integer feasible solutions found using polishing/rounding
            #
            # Initialization
            'initialization_flag': True,  # use initialization procedure
            'init_max_runtime': 120.0,  # max time to run CPA in initialization procedure
            'init_max_coefficient_gap': 0.49,
            'init_display_progress': False,  # show progress of initialization procedure
            'init_display_cplex_progress': False,  # show progress of CPLEX during intialization procedure
            #
            # CPLEX Solver Parameters
            'cplex_randomseed': self.random_state,  # random seed
            'cplex_mipemphasis': 0,  # cplex MIP strategy
        }

        data = {
            'X': X,
            'Y': y,
            'variable_names': feature_names,
            'outcome_name': "out",
            'sample_weights': np.ones(X.shape[0]),
        }

        # train model using lattice_cpa
        with redirect_stdout(open(os.devnull, 'w', encoding='utf-8')):
            model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)

        self.solution_status_code = mip_info["risk_slim_mip"].solution.get_status()

        if self.solution_status_code == 107:
            logger.warning("Solution is not optimal due to timeout")

        rho = model_info['solution']
        rho[0] = -rho[0]

        self.model = rho

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise NotFittedError()
        rho = self.model
        return expit(rho[0] - X @ rho[1:])

    def predict(self, X):
        if self.model is None:
            raise NotFittedError()
        rho = self.model
        return np.array(rho[0] <= X @ rho[1:], dtype=int)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("../data/breastcancer_processed.csv")
    X = df.loc[:, df.columns != 'Benign']
    y = df.Benign
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(RiskSlim(max_score=1, random_state=42).fit(X_train, y_train).score(X_test, y_test))
