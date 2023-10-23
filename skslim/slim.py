import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from slim_python import SLIMCoefficientConstraints, check_data, check_slim_IP_output, get_slim_summary, create_slim_IP

logger = logging.getLogger("default")


class Slim(BaseEstimator, ClassifierMixin):
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

        # run sanity checks
        feature_names = ["(Intercept)"] + [str(i) for i in range(X.shape[1] - 1)]
        check_data(X=X, Y=y, X_names=feature_names)

        #### TRAIN SCORING SYSTEM USING SLIM ####
        # setup SLIM coefficient set
        coef_constraints = SLIMCoefficientConstraints(variable_names=feature_names, ub=self.max_score,
                                                      lb=self.computed_min_score)

        # compute lower and upper bound of the bias term. excluding the bias during calculation
        # this is necessary to make sure the whole data-range can be classified
        bounds = [((y * X) * bound)[:, 1:] for bound in [self.computed_min_score, self.max_score]]

        max_scores = np.fmax(*bounds)
        max_scores = np.sum(max_scores, 1)
        intercept_lb = -max(max_scores) + 1
        coef_constraints.set_field('lb', '(Intercept)', intercept_lb)

        min_scores = np.fmin(*bounds)
        min_scores = np.sum(min_scores, 1)
        intercept_ub = -min(min_scores) + 1
        coef_constraints.set_field('ub', '(Intercept)', intercept_ub)

        # create SLIM IP
        slim_input = {
            'X': X,
            'X_names': feature_names,
            'Y': y,
            'C_0': self.C,
            'w_pos': 1.0,
            'w_neg': 1.0,
            'L0_min': 0,
            'L0_max': float('inf'),
            'err_min': 0,
            'err_max': 1.0,
            'pos_err_min': 0,
            'pos_err_max': 1.0,
            'neg_err_min': 0,
            'neg_err_max': 1.0,
            'coef_constraints': coef_constraints
        }

        model, slim_info = create_slim_IP(slim_input)

        # setup SLIM IP parameters
        # see docs/usrccplex.pdf for more about these parameters
        model.parameters.timelimit.set(self.timeout)  # set runtime here
        model.parameters.randomseed.set(self.random_state)
        model.parameters.threads.set(1)
        model.parameters.parallel.set(1)
        model.parameters.output.clonelog.set(0)
        model.parameters.mip.tolerances.mipgap.set(np.finfo(np.cfloat).eps)
        model.parameters.mip.tolerances.absmipgap.set(np.finfo(np.cfloat).eps)
        model.parameters.mip.tolerances.integrality.set(np.finfo(np.cfloat).eps)
        model.parameters.emphasis.mip.set(1)
        model.set_results_stream(None)
        model.set_log_stream(None)

        # solve SLIM IP
        try:
            model.solve()

            # run quick and dirty tests to make sure that IP output is correct
            check_slim_IP_output(model, slim_info, X, y, coef_constraints)
            summary = get_slim_summary(model, slim_info, X, y)
            self.solution_status_code = summary["solution_status_code"]

            if self.solution_status_code == 107:
                logger.warning("Solution is not optimal due to timeout")

            rho = summary["rho"]
            # inverting the intersept
            rho[0] = -rho[0]
        except ValueError:
            logger.warning("No features have been selected")
            rho = np.zeros(X.shape[1])
            rho[0] = 1

        self.model = rho

        return self

    def predict(self, X):
        if self.model is None:
            raise NotFittedError()
        rho = self.model

        return np.array(X @ rho[1:] >= rho[0], dtype=int)

    def __str__(self):
        return np.array_str(self.model)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("../data/breastcancer_processed.csv")
    X = df.loc[:, df.columns != 'Benign']
    y = df.Benign
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print(Slim(max_score=1, random_state=42).fit(X_train, y_train).score(X_test, y_test))
