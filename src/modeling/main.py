from modeling.models.first_model import run_simulation
from modeling.models.pydantic_models import Parameters


if __name__ == "__main__":

    parameters = Parameters(
        days=365 * 2,                    # horizon
        initial_supply=100_000_000.0,    # total minted at t0
        initial_circulating=10_000_000.0,# circulating at t0 (rest locked/treasury/etc.)
        initial_treasury_tokens=20_000_000.0,
        initial_treasury_quote=2_000_000.0,   # quote currency (e.g., USDC)
        # AMM reserves (token, quote) to seed price discovery
        amm_token_reserve=5_000_000.0,
        amm_quote_reserve=1_000_000.0,
        # Emissions
        daily_emission_tokens=50_000.0,       # per day, before splits
        staking_reward_split=0.6,             # % of emissions to stakers
        treasury_reward_split=0.2,            # % of emissions to treasury (token form)
        ecosystem_reward_split=0.2,           # % of emissions to ecosystem/unlocked circulation
        # Staking
        target_staked_ratio=0.35,             # fraction of circulating tokens staked (mean reversion target)
        stake_mean_reversion=0.15,            # speed of reversion
        # Vesting schedules: list of dicts with fields
        # - name, total_tokens, start_day, cliff_days, vest_days (linear unlock after cliff)
        vesting_schedules=[
            {"name": "Team", "total_tokens": 25_000_000.0, "start_day": 0, "cliff_days": 180, "vest_days": 730},
            {"name": "Investors", "total_tokens": 15_000_000.0, "start_day": 0, "cliff_days": 90, "vest_days": 365},
            {"name": "Advisors", "total_tokens": 5_000_000.0, "start_day": 0, "cliff_days": 180, "vest_days": 365},
        ],
        # AMM fee and treasury policy
        amm_fee=0.003,                   # 0.3% fee
        treasury_fee_cut=1.0,            # entire LP fee goes to treasury (protocol-owned LP assumption)
        # Treasury spending policy (runway modeling)
        treasury_monthly_burn_quote=50_000.0,  # ops spend per month in quote
        treasury_buyback_threshold=1_500_000.0, # if quote > threshold, buy back tokens worth 10k per day
        treasury_daily_buyback_quote=10_000.0,
        # Demand / order-flow process
        order_flow_sigma=75_000.0,       # daily net token buy/sell size std dev
        order_flow_ar1=0.15,             # trending behavior in net flow
        order_flow_drift=0.0,            # mean net flow; positive buys token, negative sells token
        random_seed=42,

    )

    df = run_simulation(parameters.model_dump())

    print(df)