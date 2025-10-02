"""
Tokenomics simulation: minimal, extensible framework.

Core components:
- Discrete-time daily simulation over a configurable horizon.
- Constant-product AMM for price discovery and execution.
- Emissions with configurable splits (stakers, treasury, ecosystem).
- Linear vesting schedules with cliffs for multiple cohorts.
- Simple stochastic order-flow (AR(1) shocks) driving net demand.
- Treasury policy that accrues AMM fees and optionally performs buybacks,
  while burning a fixed monthly operating spend from quote reserves.
- Staking dynamics that mean-revert toward a target staked ratio.

This module is written to be library-friendly: the main simulation logic
is in `run_simulation()`. The bottom section under `if __name__ == "__main__":`
is a demo that plots outputs and saves a CSV for exploration.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
PARAMS = {
    "days": 365 * 2,                    # horizon
    "initial_supply": 100_000_000.0,    # total minted at t0
    "initial_circulating": 10_000_000.0,# circulating at t0 (rest locked/treasury/etc.)
    "initial_treasury_tokens": 20_000_000.0,
    "initial_treasury_quote": 2_000_000.0,   # quote currency (e.g., USDC)
    # AMM reserves (token, quote) to seed price discovery
    "amm_token_reserve": 5_000_000.0,
    "amm_quote_reserve": 1_000_000.0,
    # Emissions
    "daily_emission_tokens": 50_000.0,       # per day, before splits
    "staking_reward_split": 0.6,             # % of emissions to stakers
    "treasury_reward_split": 0.2,            # % of emissions to treasury (token form)
    "ecosystem_reward_split": 0.2,           # % of emissions to ecosystem/unlocked circulation
    # Staking
    "target_staked_ratio": 0.35,             # fraction of circulating tokens staked (mean reversion target)
    "stake_mean_reversion": 0.15,            # speed of reversion
    # Vesting schedules: list of dicts with fields
    # - name, total_tokens, start_day, cliff_days, vest_days (linear unlock after cliff)
    "vesting_schedules": [
        {"name": "Team", "total_tokens": 25_000_000.0, "start_day": 0, "cliff_days": 180, "vest_days": 730},
        {"name": "Investors", "total_tokens": 15_000_000.0, "start_day": 0, "cliff_days": 90, "vest_days": 365},
        {"name": "Advisors", "total_tokens": 5_000_000.0, "start_day": 0, "cliff_days": 180, "vest_days": 365},
    ],
    # AMM fee and treasury policy
    "amm_fee": 0.003,                   # 0.3% fee
    "treasury_fee_cut": 1.0,            # entire LP fee goes to treasury (protocol-owned LP assumption)
    # Treasury spending policy (runway modeling)
    "treasury_monthly_burn_quote": 50_000.0,  # ops spend per month in quote
    "treasury_buyback_threshold": 1_500_000.0, # if quote > threshold, buy back tokens worth 10k per day
    "treasury_daily_buyback_quote": 10_000.0,
    # Demand / order-flow process
    "order_flow_sigma": 75_000.0,       # daily net token buy/sell size std dev
    "order_flow_ar1": 0.15,             # trending behavior in net flow
    "order_flow_drift": 0.0,            # mean net flow; positive buys token, negative sells token
    "random_seed": 42,
}

# -----------------------------
# Utilities
# -----------------------------

@dataclass
class VestingSchedule:
    name: str
    total_tokens: float
    start_day: int
    cliff_days: int
    vest_days: int
    unlocked_so_far: float = 0.0

    def unlock_today(self, day: int) -> float:
        """Compute tokens unlocked on a given day and update internal state.

        Unlocking is linear from the end of the cliff to the end of the vesting
        period. Prior to cliff end, zero tokens unlock. After vest end, any
        remaining balance is released in full.

        Args:
            day: Simulation day (0-indexed).

        Returns:
            The number of tokens unlocked today (non-negative float).
        """
        # before start or before cliff, no unlocks
        if day < self.start_day + self.cliff_days:
            return 0.0
        # linear unlock from cliff end to vest end
        vest_start = self.start_day + self.cliff_days
        vest_end = vest_start + self.vest_days
        if day >= vest_end:
            unlockable = self.total_tokens - self.unlocked_so_far
            self.unlocked_so_far = self.total_tokens
            return max(0.0, unlockable)
        # linear fraction
        total_linear = self.total_tokens
        elapsed = day - vest_start + 1
        frac = min(1.0, max(0.0, elapsed / self.vest_days))
        target_unlocked = total_linear * frac
        delta = target_unlocked - self.unlocked_so_far
        self.unlocked_so_far += max(0.0, delta)
        return max(0.0, delta)

class ConstantProductAMM:
    """Simple constant-product AMM (x*y=k) with fees.

    Attributes:
        token_reserve: Reserve of the simulation token (x).
        quote_reserve: Reserve of the quote currency (y), e.g., USDC.
        invariant_k: Product invariant k = x * y.
        fee: Proportional fee applied to incoming amount (e.g., 0.003 for 0.3%).
        cum_volume_quote: Cumulative volume tracked in quote terms for metrics.
    """

    def __init__(self, token_reserve: float, quote_reserve: float, fee: float):
        self.token_reserve = token_reserve
        self.quote_reserve = quote_reserve
        self.invariant_k = self.token_reserve * self.quote_reserve
        self.fee = fee
        self.cum_volume_quote = 0.0

    @property
    def price(self) -> float:
        """Implied price in quote per token (y/x)."""
        return self.quote_reserve / self.token_reserve if self.token_reserve > 0 else np.inf

    def trade_token_for_quote(self, token_in: float) -> float:
        """Swap tokens for quote, returning quote received after slippage.

        Args:
            token_in: Gross tokens user provides.

        Returns:
            quote_out: Quote amount received by the trader.
        """
        # token_in after fee
        token_in_net = token_in * (1.0 - self.fee)
        x_new = self.token_reserve + token_in_net
        y_new = self.invariant_k / x_new
        quote_out = self.quote_reserve - y_new
        # update reserves
        self.token_reserve += token_in_net
        self.quote_reserve -= quote_out
        self.invariant_k = self.token_reserve * self.quote_reserve
        self.cum_volume_quote += abs(quote_out)
        # protocol fee in quote terms can be approximated via effective price * fee component
        return quote_out

    def trade_quote_for_token(self, quote_in: float) -> float:
        """Swap quote for tokens, returning tokens received after slippage.

        Args:
            quote_in: Gross quote currency amount user provides.

        Returns:
            token_out: Token amount received by the trader.
        """
        quote_in_net = quote_in * (1.0 - self.fee)
        y_new = self.quote_reserve + quote_in_net
        x_new = self.invariant_k / y_new
        token_out = self.token_reserve - x_new
        self.quote_reserve += quote_in_net
        self.token_reserve -= token_out
        self.invariant_k = self.token_reserve * self.quote_reserve
        self.cum_volume_quote += abs(quote_in)
        return token_out

# -----------------------------
# Simulation
# -----------------------------

def run_simulation(params: Any) -> pd.DataFrame:
    """Run the tokenomics simulation for the configured number of days.

    Key steps per day:
    1) Mint emissions and split between stakers, treasury (in token), and ecosystem.
    2) Unlock vested tokens according to schedules.
    3) Mean-revert staking participation toward the target ratio.
    4) Apply order-flow shock via AMM trades (net buy/sell pressure).
    5) Apply treasury policy (monthly spend and optional buybacks).

    Args:
        params: Either a plain dict of parameters (see `PARAMS`) or a
            Pydantic `Parameters` instance providing a `.model_dump()` method.

    Returns:
        A pandas DataFrame with daily metrics over the simulation horizon.
    """
    # Accept both dict-like and Pydantic models (duck typing on model_dump)
    if hasattr(params, "model_dump"):
        params = params.model_dump()

    random_generator = np.random.default_rng(params["random_seed"])
    num_days = params["days"]
    amm = ConstantProductAMM(
        token_reserve=params["amm_token_reserve"],
        quote_reserve=params["amm_quote_reserve"],
        fee=params["amm_fee"],
    )

    # Normalize vesting schedules: accept dicts or Pydantic models with `.model_dump()`
    vesting_schedules_input = params.get("vesting_schedules", [])
    normalized_schedules: List[Dict[str, float]] = []
    for item in vesting_schedules_input:
        if hasattr(item, "model_dump"):
            normalized_schedules.append(item.model_dump())
        elif isinstance(item, dict):
            normalized_schedules.append(item)
        else:
            raise TypeError(
                "Each vesting_schedules item must be a dict or Pydantic model with .model_dump()"
            )

    vesting_schedules = [VestingSchedule(**schedule) for schedule in normalized_schedules]

    total_supply = params["initial_supply"]
    circulating = params["initial_circulating"]
    locked = total_supply - circulating

    treasury_token = params["initial_treasury_tokens"]
    treasury_quote = params["initial_treasury_quote"]

    staked = circulating * params["target_staked_ratio"]
    free_float = max(0.0, circulating - staked)

    # Order flow AR(1)
    previous_net_flow = 0.0

    daily_records: List[Dict] = []

    for day in range(num_days):
        # 1) Emissions split
        emission = params["daily_emission_tokens"]
        to_stakers = emission * params["staking_reward_split"]
        to_treasury = emission * params["treasury_reward_split"]
        to_ecosystem = emission * params["ecosystem_reward_split"]

        total_supply += emission
        treasury_token += to_treasury
        circulating += to_ecosystem + to_stakers  # staker rewards count as circulating, but are staked immediately
        staked += to_stakers

        # 2) Vesting unlocks add to circulating supply and free float (assuming recipients eventually sell or hold)
        unlocked_today = 0.0
        for schedule in vesting_schedules:
            unlocked_today += schedule.unlock_today(day)
        locked = max(0.0, locked - unlocked_today)
        circulating += unlocked_today
        free_float += unlocked_today

        # 3) Staking participation mean reversion towards target ratio
        target_staked = circulating * params["target_staked_ratio"]
        staked += params["stake_mean_reversion"] * (target_staked - staked)
        staked = max(0.0, min(staked, circulating))
        free_float = max(0.0, circulating - staked)

        # 4) Order-flow shock (net demand): positive means net buy (quote -> token), negative means net sell (token -> quote)
        order_flow_shock = (
            params["order_flow_drift"]
            + params["order_flow_ar1"] * previous_net_flow
            + random_generator.normal(0.0, params["order_flow_sigma"])
        )
        previous_net_flow = order_flow_shock

        # Cap by free float for selling pressure
        net_token_demand = order_flow_shock  # measured in tokens per day (approx)
        # Convert order flow in tokens to a trade on AMM by using current price to scale quote orders where needed.
        price_before = amm.price if amm.price and np.isfinite(amm.price) else 0.0

        protocol_fee_quote = 0.0

        if net_token_demand > 0:  # buyers bring quote to AMM to acquire tokens
            # translate token demand to a quote order with slippage on AMM by iteratively approximating
            # use one-shot: quote_in ~= demand_tokens * price_before (rough)
            quote_in = net_token_demand * price_before
            token_out = amm.trade_quote_for_token(quote_in)
            # protocol fee collection in quote approximated by fee * quote_in * treasury cut
            protocol_fee_quote = params["amm_fee"] * quote_in * params["treasury_fee_cut"]
            treasury_quote += protocol_fee_quote
            # circulation unchanged by AMM trades; only reserves change
        else:  # sellers dump tokens to AMM
            sell_tokens = min(free_float * 0.5, abs(net_token_demand))  # cannot sell more than free float share
            quote_out = amm.trade_token_for_quote(sell_tokens)
            protocol_fee_quote = params["amm_fee"] * (sell_tokens * price_before) * params["treasury_fee_cut"]
            treasury_quote += protocol_fee_quote
            # selling reduces free float (they swapped tokens for quote), but total circulating unchanged
            free_float = max(0.0, free_float - sell_tokens)

        price_after = amm.price if np.isfinite(amm.price) else price_before

        # 5) Treasury policy: spend and optional buybacks
        # monthly spend in quote
        if day % 30 == 0 and day > 0:
            spend = params["treasury_monthly_burn_quote"]
            treasury_quote = max(0.0, treasury_quote - spend)

        # buybacks if treasury rich in quote
        if treasury_quote > params["treasury_buyback_threshold"]:
            # buy a fixed amount of quote worth of tokens, move to treasury_token (optional burn)
            spend_quote = min(params["treasury_daily_buyback_quote"], treasury_quote * 0.1)
            if spend_quote > 0 and np.isfinite(amm.price) and amm.price > 0:
                token_bought = amm.trade_quote_for_token(spend_quote)
                treasury_quote -= spend_quote
                treasury_token += token_bought
                # optionally burn: uncomment to burn bought tokens
                # total_supply -= token_bought

        market_cap = price_after * circulating
        fully_diluted_value = price_after * total_supply
        staked_ratio = staked / circulating if circulating > 0 else 0.0
        runway_months = (treasury_quote / max(1.0, params["treasury_monthly_burn_quote"])) if params["treasury_monthly_burn_quote"] > 0 else np.inf

        daily_records.append({
            "day": day,
            "price": price_after,
            "circulating": circulating,
            "total_supply": total_supply,
            "staked": staked,
            "staked_ratio": staked_ratio,
            "free_float": free_float,
            "locked": locked,
            "treasury_token": treasury_token,
            "treasury_quote": treasury_quote,
            "market_cap": market_cap,
            "fdv": fully_diluted_value,
            "amm_token_reserve": amm.token_reserve,
            "amm_quote_reserve": amm.quote_reserve,
            "amm_volume_cum_quote": amm.cum_volume_quote,
        })

    results_df = pd.DataFrame(daily_records)
    return results_df

if __name__ == "__main__":
    # Demo run for interactive exploration
    results_df = run_simulation(PARAMS)

    # Display a quick dashboard
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Tokenomics Simulation (daily)", results_df.head(30))

    # Plots
    plt.figure()
    plt.plot(results_df["day"], results_df["price"])
    plt.title("Token Price (AMM implied)")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.show()

    plt.figure()
    plt.plot(results_df["day"], results_df["circulating"])
    plt.title("Circulating Supply")
    plt.xlabel("Day")
    plt.ylabel("Tokens")
    plt.show()

    plt.figure()
    plt.plot(results_df["day"], results_df["treasury_quote"])
    plt.title("Treasury Quote Balance")
    plt.xlabel("Day")
    plt.ylabel("Quote Units")
    plt.show()

    # KPIs at end
    kpis = {
        "final_price": results_df["price"].iloc[-1],
        "final_circulating": results_df["circulating"].iloc[-1],
        "final_total_supply": results_df["total_supply"].iloc[-1],
        "final_staked_ratio": results_df["staked_ratio"].iloc[-1],
        "final_market_cap": results_df["market_cap"].iloc[-1],
        "final_fdv": results_df["fdv"].iloc[-1],
        "final_treasury_quote": results_df["treasury_quote"].iloc[-1],
        "final_treasury_token": results_df["treasury_token"].iloc[-1],
        "cum_amm_volume_quote": results_df["amm_volume_cum_quote"].iloc[-1],
        "runway_months_est": (results_df["treasury_quote"].iloc[-1] / max(1.0, PARAMS["treasury_monthly_burn_quote"])) if PARAMS["treasury_monthly_burn_quote"] > 0 else float('inf')
    }
    kpis_df = pd.DataFrame([kpis])
    display_dataframe_to_user("KPIs (end of simulation)", kpis_df)

    # Save CSV for further analysis
    csv_path = "/mnt/data/tokenomics_simulation.csv"
    results_df.to_csv(csv_path, index=False)
    csv_path
