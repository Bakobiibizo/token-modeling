"""Repeatable ensemble and one-at-a-time sensitivity analysis."""
from collections.abc import Iterable, Mapping
import pandas as pd

from .models.first_model import run_simulation
from .models.pydantic_models import Parameters

DEFAULT_METRICS = ("price", "circulating", "staked_ratio", "treasury_quote")


def simulate_ensemble(parameters: Parameters, seeds: Iterable[int], metrics=DEFAULT_METRICS) -> pd.DataFrame:
    """Run identical assumptions across seeds and return terminal observations."""
    rows = []
    for seed in seeds:
        candidate = parameters.model_copy(update={"random_seed": seed})
        terminal = run_simulation(candidate).iloc[-1]
        rows.append({"seed": seed, **{metric: float(terminal[metric]) for metric in metrics}})
    return pd.DataFrame(rows)


def quantiles(ensemble: pd.DataFrame, probabilities=(0.05, 0.5, 0.95)) -> pd.DataFrame:
    """Summarize numeric ensemble outputs without implying forecasts."""
    return ensemble.drop(columns=["seed"], errors="ignore").quantile(probabilities)


def sensitivity(parameters: Parameters, changes: Mapping[str, Iterable[float]], metric="price") -> pd.DataFrame:
    """One-at-a-time terminal sensitivity around a fixed baseline seed."""
    rows = []
    for parameter, values in changes.items():
        if parameter not in Parameters.model_fields:
            raise KeyError(f"unknown parameter: {parameter}")
        for value in values:
            candidate = parameters.model_copy(update={parameter: value})
            candidate = Parameters.model_validate(candidate.model_dump())
            result = run_simulation(candidate)
            rows.append({"parameter": parameter, "value": value, "metric": metric, "terminal": float(result[metric].iloc[-1])})
    return pd.DataFrame(rows)
