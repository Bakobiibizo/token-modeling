# token-modeling

A deterministic Python library for exploring token-economy assumptions. It models daily emissions, cliff/linear vesting, staking participation, constant-product AMM execution, stochastic demand, and treasury spending/buybacks.

This is a scenario-analysis tool, not a price forecast. Results depend entirely on the stated assumptions; the demand process is deliberately synthetic and does not estimate real adoption, liquidity-provider behavior, adversarial trading, regulatory effects, or market microstructure.

## Install and run

```bash
uv sync --extra dev
uv run pytest
uv run token-model examples/baseline.json --output simulation.csv
```

Library use:

```python
from modeling.analysis import quantiles, simulate_ensemble
from modeling.models.first_model import run_simulation
from modeling.models.pydantic_models import Parameters

assumptions = Parameters(days=365, random_seed=42)
daily = run_simulation(assumptions)
distribution = quantiles(simulate_ensemble(assumptions, range(100)))
```

`Parameters` is the formal schema and validates supplies, AMM liquidity, emission splits, rates, and schedules. A fixed parameter set and seed always produces the same result. `sensitivity()` performs one-at-a-time parameter sweeps; `simulate_ensemble()` varies seeds and `quantiles()` reports empirical uncertainty bands.

## Methodology and accounting

Each daily step mints emissions, unlocks vesting, adjusts stake toward a target, executes net synthetic order flow against an `x*y=k` AMM, then applies treasury policy. Emissions increase total supply. Staker and ecosystem emissions plus vesting increase circulating supply; treasury emissions remain treasury-held. AMM trades alter reserves but not circulating supply. Protocol fees and operating spend alter treasury quote holdings.

The model is validated against known cases: zero flow preserves AMM price, fee-free swaps preserve the constant product, vesting honors its cliff and completes exactly, supply grows by emissions, and repeated seeded runs match byte-for-byte.

The optional web UI is intentionally outside the core dependency boundary because it uses Greeble:

```bash
uv sync --extra web
uv run uvicorn modeling.web.app:app
```

The old `token_modeling/` directory is a historical Greeble starter snapshot and is not packaged. Production integrations should import only `modeling` from `src/`.
