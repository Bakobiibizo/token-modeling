import pytest

from modeling.analysis import quantiles, sensitivity, simulate_ensemble
from modeling.models.first_model import ConstantProductAMM, VestingSchedule, run_simulation
from modeling.models.pydantic_models import Parameters


def test_seeded_simulation_is_deterministic_and_accounts_for_emissions():
    p = Parameters(days=10, vesting_schedules=[])
    left, right = run_simulation(p), run_simulation(p)
    assert left.equals(right)
    assert left.iloc[-1].total_supply == p.initial_supply + p.days * p.daily_emission_tokens
    assert (left.select_dtypes("number") >= 0).all().all()


def test_zero_flow_known_case_keeps_amm_price_constant():
    p = Parameters(days=5, vesting_schedules=[], order_flow_sigma=0, order_flow_drift=0,
                   treasury_buyback_threshold=1e20)
    result = run_simulation(p)
    assert result.price.tolist() == pytest.approx([p.amm_quote_reserve / p.amm_token_reserve] * 5)


def test_constant_product_and_known_swap_direction():
    amm = ConstantProductAMM(1000, 1000, 0)
    initial_k = amm.invariant_k
    assert amm.trade_quote_for_token(100) > 0
    assert amm.price > 1
    assert amm.invariant_k == pytest.approx(initial_k)


def test_vesting_cliff_and_completion():
    schedule = VestingSchedule("team", 100, 0, 2, 2)
    assert [schedule.unlock_today(day) for day in range(4)] == [0, 0, 50, 50]


def test_ensemble_quantiles_and_sensitivity():
    p = Parameters(days=4, vesting_schedules=[])
    ensemble = simulate_ensemble(p, [1, 2, 3])
    assert ensemble.seed.tolist() == [1, 2, 3]
    assert quantiles(ensemble).shape == (3, 4)
    result = sensitivity(p, {"order_flow_sigma": [0, 10]})
    assert result.shape[0] == 2


def test_invalid_accounting_rejected():
    with pytest.raises(ValueError):
        Parameters(initial_supply=1, initial_circulating=2)
