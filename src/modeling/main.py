"""Command-line interface for reproducible scenario runs."""
import argparse
import json
from pathlib import Path

from .models.first_model import run_simulation
from .models.pydantic_models import Scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="JSON file matching the Scenario schema")
    parser.add_argument("--output", type=Path, default=Path("simulation.csv"))
    args = parser.parse_args()
    scenario = Scenario.model_validate(json.loads(args.scenario.read_text()))
    result = run_simulation(scenario.parameters)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result)} days for {scenario.name!r} to {args.output}")


if __name__ == "__main__":
    main()
