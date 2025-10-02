from pydantic import BaseModel, Field, model_validator


class VestingSchedule(BaseModel):
    """Typed vesting schedule configuration for a cohort.

    Unlocks linearly after a cliff for a specified number of days.
    """

    name: str = Field(..., description="Cohort name (e.g., Team, Investors)")
    total_tokens: float = Field(..., ge=0, description="Total tokens allocated to this cohort")
    start_day: int = Field(0, ge=0, description="Start day for vesting schedule (0-indexed)")
    cliff_days: int = Field(0, ge=0, description="Cliff duration in days before any unlocks")
    vest_days: int = Field(..., ge=0, description="Linear vesting duration in days after cliff")


class Parameters(BaseModel):
    """Simulation parameter set with validation and sensible defaults.

    This mirrors the legacy PARAMS dict but provides type safety and validation.
    """

    # Horizon and initial supplies
    days: int = Field(365 * 2, ge=1, description="Simulation horizon in days")
    initial_supply: float = Field(100_000_000.0, ge=0)
    initial_circulating: float = Field(10_000_000.0, ge=0)
    initial_treasury_tokens: float = Field(20_000_000.0, ge=0)
    initial_treasury_quote: float = Field(2_000_000.0, ge=0)

    # AMM seeding
    amm_token_reserve: float = Field(5_000_000.0, ge=0)
    amm_quote_reserve: float = Field(1_000_000.0, ge=0)

    # Emissions
    daily_emission_tokens: float = Field(50_000.0, ge=0)
    staking_reward_split: float = Field(0.6, ge=0.0, le=1.0)
    treasury_reward_split: float = Field(0.2, ge=0.0, le=1.0)
    ecosystem_reward_split: float = Field(0.2, ge=0.0, le=1.0)

    # Staking dynamics
    target_staked_ratio: float = Field(0.35, ge=0.0, le=1.0)
    stake_mean_reversion: float = Field(0.15, ge=0.0, le=1.0)

    # Vesting schedules
    vesting_schedules: list[VestingSchedule] = Field(
        default_factory=lambda: [
            VestingSchedule(name="Team", total_tokens=25_000_000.0, start_day=0, cliff_days=180, vest_days=730),
            VestingSchedule(name="Investors", total_tokens=15_000_000.0, start_day=0, cliff_days=90, vest_days=365),
            VestingSchedule(name="Advisors", total_tokens=5_000_000.0, start_day=0, cliff_days=180, vest_days=365),
        ]
    )

    # AMM fee and treasury policy
    amm_fee: float = Field(0.003, ge=0.0, le=1.0)
    treasury_fee_cut: float = Field(1.0, ge=0.0, le=1.0)

    # Treasury spending policy (runway modeling)
    treasury_monthly_burn_quote: float = Field(50_000.0, ge=0)
    treasury_buyback_threshold: float = Field(1_500_000.0, ge=0)
    treasury_daily_buyback_quote: float = Field(10_000.0, ge=0)

    # Demand / order-flow process
    order_flow_sigma: float = Field(75_000.0, ge=0)
    order_flow_ar1: float = Field(0.15, ge=-1.0, le=1.0)
    order_flow_drift: float = Field(0.0)
    random_seed: int = Field(42)

    @model_validator(mode="after")
    def validate_splits(self):
        total = self.staking_reward_split + self.treasury_reward_split + self.ecosystem_reward_split
        # Allow tiny floating error
        if not (abs(total - 1.0) <= 1e-8):
            raise ValueError(
                f"Emission splits must sum to 1.0; got {total:.8f}"
            )
        return self