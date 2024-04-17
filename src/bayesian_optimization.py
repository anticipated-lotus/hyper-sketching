from typing import Any, Callable, Dict

from hyperopt import fmin, space_eval, tpe
from hyperopt.early_stop import no_progress_loss


def hyperopt_optimization(
    objective: Callable, space: Dict[str, Callable], max_evals: int
) -> Any:
    best = fmin(
        fn=objective,  # Objective Function to optimize
        space=space,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm (representative TPE)
        max_evals=max_evals,  # Number of optimization attempts
        verbose=True,
        show_progressbar=True,
        early_stop_fn=no_progress_loss(iteration_stop_count=40, percent_increase=0.01),
    )
    best = space_eval(space, best)
    return best
