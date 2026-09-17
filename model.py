"""Integrated weather, dispatch, economics, and reporting model.

Canonical units are kW, kWh, kg, h, K, bar, and USD. The equation-level process
calculations live in ``functions.py``. This module
contains the stateful/orchestrating layer needed by the notebook and Dash app.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
import json
import math
import os
import time

import numpy as np
import pandas as pd
import requests
import functions as calculations
from ninja_usage import record_request, record_response


DEFAULT_CACHE_DIR = Path("data") / "weather_cache"
DEFAULT_OUTPUT_DIR = Path("outputs")
NINJA_API_URL = "https://www.renewables.ninja/api/data/pv"
# Renewables.ninja answers a cached year in seconds but can take minutes to compute an
# uncached one. A tight read timeout turns "slow" into "failed" and burns one quota call
# per attempt, so allow generous headroom before giving up.
NINJA_READ_TIMEOUT_SECONDS = 180
ProgressCallback = Callable[[str], None]
MAX_CAPACITY_CALIBRATION_EVALUATIONS = 25
# Exponent of the Laue air-mass attenuation correlation, tau ** (AM ** 0.678).
AIR_MASS_EXPONENT = 0.678


def _report_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


class ModelError(RuntimeError):
    """Raised when inputs cannot produce a physically meaningful model run."""


class WeatherDataError(ModelError):
    """Raised for weather download, cache, or validation failures."""


@dataclass(frozen=True)
class PlantParameters:
    """Algebraic process assumptions on a one-kilogram methane basis."""

    solar_farm_mw: float = 10.0
    electrolyser_kwh_per_kg_h2: float = 52.0
    fuel_cell_kwh_per_kg_h2: float = 18.33
    fan_kwh_per_kg_co2: float = 0.50
    sabatier_heat_kwh_per_kg_ch4: float = 2.85
    co2_outlet_pressure_bar: float = 40.0
    compressor_isentropic_efficiency: float = 0.75
    compressor_stages: int = 3
    intercool_temperature_k: float = 313.15
    sabatier_temperature_k: float = 623.15
    # Carbonation from 400 ppm air is only thermodynamically possible below about
    # 790 K; see maximum_carbonator_temperature_k on PlantEnergyResult. 673.15 K
    # keeps a 113x CO2 partial-pressure driving force while staying hot enough for
    # workable CaO carbonation kinetics.
    carbonator_temperature_k: float = 673.15
    calciner_temperature_k: float = 1123.15
    reference_ambient_temperature_k: float = 288.15
    # The single most influential parameter in the model: the DAC air stream carries
    # by far the largest heat-capacity flow, so E_req scales with (1 - this value).
    # Set explicitly on the Single site tab.
    air_exhaust_hx_effectiveness: float = 0.97
    solids_heat_recovery_efficiency: float = 0.90
    air_co2_mole_fraction: float = 400e-6
    dac_capture_efficiency: float = 0.75
    air_pressure_bar: float = 1.01325
    air_fallback_cp_kwh_per_kg_k: float = 0.000279
    air_molar_mass_kg_per_mol: float = 0.028965
    cao_cp_kwh_per_kg_k: float = 0.000278
    caco3_cp_kwh_per_kg_k: float = 0.000250
    solids_cycle_time_h: float = 4.0
    calcination_delta_h_j_per_mol: float = 178_300.0
    calcination_delta_s_j_per_mol_k: float = 160.6
    ramp_fraction_per_h: float = 1.0
    methane_molar_mass_kg_per_mol: float = 0.0160425
    hydrogen_molar_mass_kg_per_mol: float = 0.00201588
    carbon_dioxide_molar_mass_kg_per_mol: float = 0.0440095
    water_molar_mass_kg_per_mol: float = 0.01801528
    calcium_oxide_molar_mass_kg_per_mol: float = 0.056077
    calcium_carbonate_molar_mass_kg_per_mol: float = 0.1000869
    gas_constant_j_per_mol_k: float = 8.314462618
    hydrogen_lhv_kwh_per_kg: float = 33.33
    co2_compression_fallback_cp_j_per_kg_k: float = 846.0
    co2_compression_fallback_gamma: float = 1.289
    co2_sensible_fallback_cp_j_per_kg_k: float = 900.0
    h2_sensible_fallback_cp_j_per_kg_k: float = 14_300.0
    hydrogen_storage_pressure_bar: float = 300.0
    co2_storage_pressure_bar: float = 150.0
    methane_storage_pressure_bar: float = 300.0
    methane_delivery_pressure_bar: float = 1.0
    storage_expander_isentropic_efficiency: float = 0.80
    storage_machine_stages: int = 3
    hydrogen_compression_fallback_gamma: float = 1.41
    methane_compression_fallback_cp_j_per_kg_k: float = 2_220.0
    methane_compression_fallback_gamma: float = 1.31

    def __post_init__(self) -> None:
        if not 0 < self.compressor_isentropic_efficiency <= 1:
            raise ValueError("Compressor isentropic efficiency must lie above zero and at or below one")
        if not 0 < self.storage_expander_isentropic_efficiency <= 1:
            raise ValueError("Storage expander isentropic efficiency must lie above zero and at or below one")
        if self.compressor_stages < 1 or self.storage_machine_stages < 1:
            raise ValueError("Compressor and storage-machine stage counts must be positive")
        if self.co2_outlet_pressure_bar <= 0 or self.methane_delivery_pressure_bar <= 0:
            raise ValueError("Gas connection pressures must be positive")
        if not 0 < self.dac_capture_efficiency <= 1:
            raise ValueError("dac_capture_efficiency must lie above zero and at or below one")
        if self.hydrogen_storage_pressure_bar <= self.co2_outlet_pressure_bar:
            raise ValueError("Hydrogen storage pressure must exceed the process pressure")
        if self.co2_storage_pressure_bar <= self.co2_outlet_pressure_bar:
            raise ValueError("CO2 storage pressure must exceed the process pressure")
        if self.methane_storage_pressure_bar <= self.co2_outlet_pressure_bar:
            raise ValueError("Methane storage pressure must exceed the process pressure")
        if self.methane_storage_pressure_bar <= self.methane_delivery_pressure_bar:
            raise ValueError("Methane storage pressure must exceed the delivery pressure")
        if any(value <= 0 for value in (
            self.co2_compression_fallback_cp_j_per_kg_k,
            self.co2_compression_fallback_gamma - 1,
            self.h2_sensible_fallback_cp_j_per_kg_k,
            self.hydrogen_compression_fallback_gamma - 1,
            self.methane_compression_fallback_cp_j_per_kg_k,
            self.methane_compression_fallback_gamma - 1,
        )):
            raise ValueError("Gas storage fallback properties must be positive and gamma must exceed one")


@dataclass(frozen=True)
class ThermalParameters:
    """Lumped thermal-model parameters for common units and Sabatier trains."""

    calciner_ua_kw_per_k: float = 0.10
    carbonator_ua_kw_per_k: float = 0.10
    fixed_hot_auxiliary_kw: float = 25.0
    ambient_fallback_k: float = 288.15
    sabatier_reference_capacity_kg_ch4_h: float = 100.0
    sabatier_reference_thermal_capacity_kwh_per_k: float = 1.0
    sabatier_reference_ua_kw_per_k: float = 0.10
    sabatier_ua_scaling_exponent: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        positive = (
            self.sabatier_reference_capacity_kg_ch4_h,
            self.sabatier_reference_thermal_capacity_kwh_per_k,
            self.sabatier_reference_ua_kw_per_k,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Sabatier reference capacity thermal capacity and UA must be positive")
        if not 0 < self.sabatier_ua_scaling_exponent <= 1:
            raise ValueError("sabatier_ua_scaling_exponent must lie above 0 and at or below 1")


@dataclass(frozen=True)
class StorageParameters:
    """Storage capacity is battery energy or stored-hydrogen LHV energy.

    ``h2_co2`` adds a separate compressed-CO2 mass inventory.  Hydrogen remains
    the carrier-energy basis so existing storage reports and f_SOCP semantics stay
    comparable across the three long-duration options.
    """

    method: Literal["battery", "hydrogen", "h2_co2"] = "battery"
    capacity_kwh: float = 0.0
    co2_capacity_kg: float = 0.0
    max_charge_kw: float = math.inf
    max_discharge_kw: float = math.inf
    hydrogen_lhv_kwh_per_kg: float = 33.33
    hydrogen_electrolyser_kwh_per_kg: float = 52.0
    hydrogen_fuel_cell_kwh_per_kg: float = 18.33
    co2_to_hydrogen_mass_ratio: float = 5.46
    co2_production_kwh_per_kg: float = 2.0
    co2_compression_kwh_per_kg: float = 0.1
    hydrogen_compression_kwh_per_kg: float = 0.0
    hydrogen_expansion_kwh_per_kg: float = 0.0
    co2_expansion_kwh_per_kg: float = 0.0
    hydrogen_expansion_reheat_kwh_per_kg: float = 0.0
    co2_expansion_reheat_kwh_per_kg: float = 0.0
    hydrogen_charge_capacity_kw: float = 0.0
    hydrogen_fuel_cell_capacity_kw: float = 0.0
    hydrogen_storage_compressor_capacity_kw: float = 0.0
    hydrogen_storage_expander_capacity_kw: float = 0.0
    co2_charge_compressor_capacity_kw: float = 0.0
    co2_storage_expander_capacity_kw: float = 0.0
    self_discharge_fraction_per_h: float = 0.0
    co2_self_discharge_fraction_per_h: float = 0.0
    initial_soc_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_soc_fraction <= 1.0:
            raise ValueError("initial_soc_fraction must lie between 0 and 1")
        if self.method in {"hydrogen", "h2_co2"} and (
            self.hydrogen_lhv_kwh_per_kg <= 0
            or self.hydrogen_electrolyser_kwh_per_kg <= 0
            or self.hydrogen_fuel_cell_kwh_per_kg <= 0
            or self.hydrogen_compression_kwh_per_kg < 0
            or self.hydrogen_expansion_kwh_per_kg < 0
            or self.hydrogen_expansion_reheat_kwh_per_kg < 0
        ):
            raise ValueError("Hydrogen conversion intensities must be positive")
        # Discharge divides by this net intensity, so it must stay strictly positive
        # once the expander's electrical reheat load is deducted.
        if self.method in {"hydrogen", "h2_co2"} and (
            self.hydrogen_fuel_cell_kwh_per_kg
            + self.hydrogen_expansion_kwh_per_kg
            - self.hydrogen_expansion_reheat_kwh_per_kg <= 0
        ):
            raise ValueError(
                "Hydrogen expander reheat cannot cancel the net discharge output"
            )
        if self.method == "h2_co2" and (
            self.co2_capacity_kg < 0
            or self.co2_to_hydrogen_mass_ratio <= 0
            or self.co2_production_kwh_per_kg <= 0
            or self.co2_compression_kwh_per_kg <= 0
            or self.co2_expansion_kwh_per_kg < 0
            or self.co2_expansion_reheat_kwh_per_kg < 0
        ):
            raise ValueError("H2 + CO2 storage requires positive CO2 sizing and production parameters")
        if not 0.0 <= self.co2_self_discharge_fraction_per_h < 1.0:
            raise ValueError("co2_self_discharge_fraction_per_h must lie at or above 0 and below 1")


@dataclass(frozen=True)
class StrategyConfig:
    short_strategy: Literal["through_night", "limping", "hard_shutdown"] = "limping"
    # Constant output is the only long-horizon policy in the current prototype.
    long_strategy: Literal["constant_output"] = "constant_output"
    f_ocp: float = 0.20
    # Long-term installed/required energy-capacity ratio (1.0 = exactly sized).
    # Short-term storage is always installed at its calculated daily-cycle requirement.
    f_socp_long: float = 1.0
    daylight_cf_cutoff: float = 0.001
    parallel_reactor_count: int = 1
    reactor_scheduling_mode: Literal["daily_storage_aware", "seasonal"] = "seasonal"
    # Ten days to match the forecast horizon the competition brief supplies.
    storage_planning_lookahead_days: int = 10
    # Whether the product vessel is this plant's seasonal store. Off — the default and
    # the case for every upstream storage method — methane leaves the gate as it is
    # made, with no vessel, no compression into it and no expansion credit out of it,
    # and f_SOCP sizes the energy store. On, it is the reverse: no upstream store is
    # built, the plant follows the sun, and f_SOCP sizes the vessel instead. The
    # dashboard derives this from the chosen long-storage method rather than offering
    # it as a switch, because building both would be two seasonal stores in series.
    product_storage: bool = False

    def __post_init__(self) -> None:
        if int(self.parallel_reactor_count) != self.parallel_reactor_count or self.parallel_reactor_count < 1:
            raise ValueError("parallel_reactor_count must be a positive integer")
        if self.reactor_scheduling_mode not in {"daily_storage_aware", "seasonal"}:
            raise ValueError("reactor_scheduling_mode must be 'daily_storage_aware' or 'seasonal'")
        if (int(self.storage_planning_lookahead_days) != self.storage_planning_lookahead_days
                or self.storage_planning_lookahead_days < 1):
            raise ValueError("storage_planning_lookahead_days must be a positive integer")


@dataclass(frozen=True)
class WeatherConfig:
    lat: float
    lon: float
    dataset: str = "merra2"
    system_loss: float = 0.10
    tracking: int = 0
    tilt: float = 35.0
    azim: float = 180.0
    latest_year: int | None = None
    training_years: int = 5
    evaluation_years: int = 10


@dataclass(frozen=True)
class SyntheticWeatherParameters:
    """Shape parameters for the deterministic offline weather profile."""

    cloud_variability: float = 0.20
    clear_sky_index: float = 0.75
    atmospheric_transmittance: float = 0.70
    axial_tilt_deg: float = 23.44
    array_tilt_deg: float = 35.0
    solar_seasonal_phase_day: float = 80.0
    seasonal_period_days: float = 365.25
    solar_noon_hour: float = 12.0
    minimum_sun_elevation_sine: float = 0.02
    maximum_capacity_factor: float = 0.95
    ambient_mean_temperature_k: float = 283.15
    ambient_seasonal_amplitude_k: float = 10.0
    ambient_seasonal_phase_day: float = 172.0
    ambient_diurnal_amplitude_k: float = 3.0
    ambient_diurnal_phase_hour: float = 14.0
    diurnal_period_hours: float = 24.0


@dataclass(frozen=True)
class EconomicParameters:
    costs_path: str = "costs.json"
    costs: dict[str, Any] | None = None


def _utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


FaultComponent = Literal["battery", "hydrogen", "sabatier", "dac"]


@dataclass(frozen=True)
class FaultDistribution:
    """Stochastic fault inputs for one subsystem.

    Capacity is the retained fraction. A zero event interval disables generation.
    """

    mean_months_between_faults: int = 6
    mean_duration_h: float = 48.0
    mean_capacity_fraction: float = 0.80

    def __post_init__(self) -> None:
        months = self.mean_months_between_faults
        if (isinstance(months, bool) or int(months) != months
                or not 0 <= months <= 120):
            raise ValueError("mean_months_between_faults must be an integer from 0 to 120")
        if (not math.isfinite(self.mean_duration_h)
                or not 1 <= self.mean_duration_h <= 720):
            raise ValueError("mean_duration_h must lie between 1 and 720")
        if (not math.isfinite(self.mean_capacity_fraction)
                or not 0.0 <= self.mean_capacity_fraction <= 1.0):
            raise ValueError("mean_capacity_fraction must lie between zero and one")


@dataclass(frozen=True)
class FaultScenario:
    battery: FaultDistribution = field(default_factory=FaultDistribution)
    hydrogen: FaultDistribution = field(default_factory=FaultDistribution)
    sabatier: FaultDistribution = field(default_factory=FaultDistribution)
    dac: FaultDistribution = field(default_factory=FaultDistribution)
    seed: int = 0

    def __post_init__(self) -> None:
        if (isinstance(self.seed, bool) or int(self.seed) != self.seed
                or not 0 <= self.seed <= np.iinfo(np.uint64).max):
            raise ValueError("seed must be an integer between 0 and 2^64 - 1")


@dataclass(frozen=True)
class FaultEvent:
    component: FaultComponent
    start: pd.Timestamp | str
    end: pd.Timestamp | str
    capacity_fraction: float

    def __post_init__(self) -> None:
        if self.component not in {"battery", "hydrogen", "sabatier", "dac"}:
            raise ValueError(f"Unsupported fault component: {self.component}")
        start_stamp = _utc_timestamp(self.start)
        end_stamp = _utc_timestamp(self.end)
        fraction = float(self.capacity_fraction)
        if end_stamp <= start_stamp:
            raise ValueError("Fault event end must be after its start")
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("Fault capacity fraction must lie between zero and one")
        object.__setattr__(self, "start", start_stamp)
        object.__setattr__(self, "end", end_stamp)
        object.__setattr__(self, "capacity_fraction", fraction)


_FAULT_STREAM_KEYS = {
    "battery": 0x42415454,
    "hydrogen": 0x48324741,
    "sabatier": 0x53414241,
    "dac": 0x44414300,
}
_FAULT_GENERATOR_VERSION = 2


def _fault_rng(seed: int, component: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), _FAULT_STREAM_KEYS[component]]))


def _generate_subsystem_faults(
    index: pd.DatetimeIndex,
    component: Literal["battery", "hydrogen", "sabatier", "dac"],
    distribution: FaultDistribution,
    seed: int,
) -> tuple[FaultEvent, ...]:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Fault generation requires a DatetimeIndex")
    if len(index) == 0 or distribution.mean_months_between_faults == 0:
        return ()
    utc_index = pd.DatetimeIndex(index).as_unit("ns")
    utc_index = utc_index.tz_localize("UTC") if utc_index.tz is None else utc_index.tz_convert("UTC")
    if not utc_index.is_monotonic_increasing or utc_index.has_duplicates:
        raise ValueError("Fault-generation timestamps must be unique and increasing")
    if len(utc_index) > 1:
        spacing = np.diff(utc_index.asi8)
        if not np.all(spacing == pd.Timedelta(hours=1).value):
            raise ValueError("Fault generation requires an uninterrupted hourly grid")

    rng = _fault_rng(seed, component)
    hours_per_month = 24.0 * 365.25 / 12.0
    probability = 1.0 - math.exp(
        -1.0 / (hours_per_month * distribution.mean_months_between_faults)
    )
    starts = np.flatnonzero(rng.random(len(utc_index)) < probability)
    events: list[FaultEvent] = []
    configured_capacity = distribution.mean_capacity_fraction
    mean_derating = 1.0 - configured_capacity
    for position in starts:
        duration = max(1, int(np.rint(rng.normal(
            distribution.mean_duration_h, 0.25 * distribution.mean_duration_h,
        ))))
        if configured_capacity in {0.0, 1.0}:
            retained = configured_capacity
        else:
            derating = float(np.clip(
                rng.normal(mean_derating, 0.25 * mean_derating), 0.0, 1.0,
            ))
            retained = 1.0 - derating
        start = utc_index[position]
        events.append(FaultEvent(
            component, start, start + pd.Timedelta(hours=duration), retained,
        ))
    return tuple(events)


def generate_battery_faults(index: pd.DatetimeIndex,
                            distribution: FaultDistribution = FaultDistribution(), *,
                            seed: int = 0) -> tuple[FaultEvent, ...]:
    return _generate_subsystem_faults(index, "battery", distribution, seed)


def generate_hydrogen_faults(index: pd.DatetimeIndex,
                             distribution: FaultDistribution = FaultDistribution(), *,
                             seed: int = 0) -> tuple[FaultEvent, ...]:
    return _generate_subsystem_faults(index, "hydrogen", distribution, seed)


def generate_sabatier_faults(index: pd.DatetimeIndex,
                             distribution: FaultDistribution = FaultDistribution(), *,
                             seed: int = 0) -> tuple[FaultEvent, ...]:
    return _generate_subsystem_faults(index, "sabatier", distribution, seed)


def generate_dac_faults(index: pd.DatetimeIndex,
                        distribution: FaultDistribution = FaultDistribution(), *,
                        seed: int = 0) -> tuple[FaultEvent, ...]:
    return _generate_subsystem_faults(index, "dac", distribution, seed)


def generate_fault_events(index: pd.DatetimeIndex,
                          scenario: FaultScenario = FaultScenario()) -> tuple[FaultEvent, ...]:
    events = (
        *generate_battery_faults(index, scenario.battery, seed=scenario.seed),
        *generate_hydrogen_faults(index, scenario.hydrogen, seed=scenario.seed),
        *generate_sabatier_faults(index, scenario.sabatier, seed=scenario.seed),
        *generate_dac_faults(index, scenario.dac, seed=scenario.seed),
    )
    return tuple(sorted(events, key=lambda event: (event.start, event.component, event.end)))


def _fault_event_dict(event: FaultEvent) -> dict[str, Any]:
    return {
        "component": event.component,
        "start": _utc_timestamp(event.start).isoformat(),
        "end": _utc_timestamp(event.end).isoformat(),
        "capacity_fraction": event.capacity_fraction,
        "duration_h": (event.end - event.start) / pd.Timedelta(hours=1),
    }


def _merged_fault_intervals(faults: Sequence[FaultEvent], *,
                            horizon: pd.DatetimeIndex | None = None) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    horizon_start = _utc_timestamp(horizon.min()) if horizon is not None and len(horizon) else None
    horizon_end = (_utc_timestamp(horizon.max()) + pd.Timedelta(hours=1)
                   if horizon is not None and len(horizon) else None)
    for component in ("battery", "hydrogen", "sabatier", "dac"):
        component_events = [event for event in faults if event.component == component]
        if not component_events:
            continue
        boundaries = sorted({
            boundary
            for event in component_events
            for boundary in (_utc_timestamp(event.start), _utc_timestamp(event.end))
        })
        for start, end in zip(boundaries, boundaries[1:]):
            clipped_start = max(start, horizon_start) if horizon_start is not None else start
            clipped_end = min(end, horizon_end) if horizon_end is not None else end
            if clipped_end <= clipped_start:
                continue
            active = [event for event in component_events
                      if _utc_timestamp(event.start) <= start < _utc_timestamp(event.end)]
            if not active:
                continue
            capacity = min(event.capacity_fraction for event in active)
            if (intervals and intervals[-1]["component"] == component
                    and intervals[-1]["end"] == clipped_start.isoformat()
                    and math.isclose(intervals[-1]["capacity_fraction"], capacity)):
                intervals[-1]["end"] = clipped_end.isoformat()
                intervals[-1]["duration_h"] = (
                    _utc_timestamp(intervals[-1]["end"])
                    - _utc_timestamp(intervals[-1]["start"])
                ) / pd.Timedelta(hours=1)
            else:
                intervals.append({
                    "component": component,
                    "start": clipped_start.isoformat(),
                    "end": clipped_end.isoformat(),
                    "capacity_fraction": capacity,
                    "duration_h": (clipped_end - clipped_start) / pd.Timedelta(hours=1),
                })
    return sorted(intervals, key=lambda item: (item["start"], item["component"]))


@dataclass(frozen=True)
class PlantEnergyResult:
    e_req_kwh_per_kg_ch4: float
    breakdown_kwh_per_kg_ch4: dict[str, float]
    stoichiometry_kg_per_kg_ch4: dict[str, float]
    calcination_equilibrium_pressure_bar: float
    compressor_kwh_per_kg_co2: float
    warnings: tuple[str, ...] = ()
    gas_storage_work_kwh_per_kg: dict[str, float] = field(default_factory=dict)
    # Carbonator feasibility margin. Carbonation needs the offered CO2 partial
    # pressure to exceed the CaO/CaCO3 equilibrium pressure, so the ratio must be
    # above one and the operating temperature below the maximum.
    carbonation_equilibrium_pressure_bar: float = 0.0
    air_co2_partial_pressure_bar: float = 0.0
    carbonation_driving_force_ratio: float = 0.0
    maximum_carbonator_temperature_k: float = 0.0


@dataclass(frozen=True)
class StorageSizingResult:
    short_capacity_kwh: float
    long_capacity_kwh: float
    short_required_charge_power_kw: float
    short_required_discharge_power_kw: float
    long_required_charge_power_kw: float
    long_required_discharge_power_kw: float
    short_initial_soc_kwh: float
    long_initial_soc_kwh: float
    nominal_methane_kg_h: float
    balance_deficit_kwh: float
    average_annual_balance_deficit_kwh: float
    feasible: bool
    warnings: tuple[str, ...] = ()
    long_co2_capacity_kg: float = 0.0
    long_initial_co2_kg: float = 0.0


@dataclass
class SimulationResult:
    hourly: pd.DataFrame
    daily: pd.DataFrame
    metrics: dict[str, float | int | str | None]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    case_id: str
    energy: PlantEnergyResult
    sizing: StorageSizingResult
    perfect: SimulationResult
    imperfect: SimulationResult | None
    economics_perfect: dict[str, Any]
    economics_imperfect: dict[str, Any] | None
    equipment_sizing: tuple[dict[str, Any], ...] = ()
    output_dir: Path | None = None
    imperfect_with_faults: SimulationResult | None = None
    economics_imperfect_with_faults: dict[str, Any] | None = None
    baseline: SimulationResult | None = None
    economics_baseline: dict[str, Any] | None = None
    # The plant the forecast-driven cases actually run, when it was sized before the
    # evaluation period rather than on it, with its own equipment register. Both are
    # empty when every case shares `sizing`. The realised deficit is that plant's
    # cyclic shortfall on the weather it was evaluated against, which its own sizing
    # study could not have seen.
    sizing_imperfect: StorageSizingResult | None = None
    equipment_sizing_imperfect: tuple[dict[str, Any], ...] = ()
    imperfect_realised_deficit_kwh_per_year: float = 0.0


@dataclass(frozen=True)
class CapacityCalibrationResult:
    strategy: StrategyConfig
    starting_factors: tuple[float, float]
    optimized_factors: tuple[float, float]
    starting_lcom_usd_per_kg_ch4: float | None
    optimized_lcom_usd_per_kg_ch4: float | None
    evaluations: int
    converged: bool
    limit_reached: bool
    feasible: bool
    average_annual_balance_deficit_kwh: float


# Weather acquisition and climatology -------------------------------------------------


def _load_api_token(env_file: str | Path = ".env") -> str | None:
    token = os.getenv("RENEWABLES_NINJA_TOKEN")
    if token:
        return token.strip()
    path = Path(env_file)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("RENEWABLES_NINJA_TOKEN="):
            return line.split("=", 1)[1].strip().strip("\"'") or None
    return None


def _weather_cache_key(config: WeatherConfig) -> str:
    payload = {
        "lat": round(config.lat, 6), "lon": round(config.lon, 6),
        "dataset": config.dataset, "system_loss": config.system_loss,
        "tracking": config.tracking, "tilt": config.tilt, "azim": config.azim,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _cache_paths(config: WeatherConfig, year: int, cache_dir: str | Path) -> tuple[Path, Path]:
    site_dir = Path(cache_dir) / _weather_cache_key(config)
    return site_dir / f"{year}.csv", site_dir / f"{year}.metadata.json"


def cached_weather_years(config: WeatherConfig, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[int]:
    site_dir = Path(cache_dir) / _weather_cache_key(config)
    if not site_dir.exists():
        return []
    return sorted(int(path.stem) for path in site_dir.glob("[0-9][0-9][0-9][0-9].csv"))


def cached_weather_sites(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[dict[str, Any]]:
    """Describe every site present in the weather cache.

    The cache key is a hash, so the coordinates are recovered from the request
    parameters each year's metadata file records.
    """
    sites: list[dict[str, Any]] = []
    root = Path(cache_dir)
    if not root.exists():
        return sites
    for site_dir in sorted(root.iterdir()):
        if not site_dir.is_dir():
            continue
        years = sorted(int(path.stem) for path in site_dir.glob("[0-9][0-9][0-9][0-9].csv"))
        if not years:
            continue
        params: dict[str, Any] = {}
        for metadata_path in sorted(site_dir.glob("[0-9][0-9][0-9][0-9].metadata.json")):
            try:
                params = json.loads(metadata_path.read_text(encoding="utf-8")).get("params") or {}
            except (OSError, ValueError):
                continue
            if params.get("lat") is not None and params.get("lon") is not None:
                break
        if params.get("lat") is None or params.get("lon") is None:
            continue
        sites.append({
            "cache_key": site_dir.name, "lat": float(params["lat"]),
            "lon": float(params["lon"]), "years": years,
            "tilt": float(params.get("tilt", 35.0)),
            "azim": float(params.get("azim", 180.0)),
            "dataset": params.get("dataset", "merra2"),
        })
    return sorted(sites, key=lambda site: -site["lat"])


def discover_latest_weather_year(
    config: WeatherConfig,
    *,
    token: str | None = None,
    start_year: int | None = None,
) -> int:
    """Probe one day per year backwards until Renewables.ninja returns PV data."""
    api_token = token or _load_api_token()
    if not api_token:
        raise WeatherDataError("A Renewables.ninja token is required to discover the latest year")
    session = requests.Session()
    session.trust_env = False
    headers = {"Authorization": f"Token {api_token}", "Accept": "application/json",
               "User-Agent": "sota-sabatier/0.1 (research prototype)"}
    first = start_year or datetime.now(UTC).year - 1
    for year in range(first, 1999, -1):
        params = {"lat": config.lat, "lon": config.lon, "date_from": f"{year}-06-01",
                  "date_to": f"{year}-06-01", "dataset": config.dataset, "capacity": 1.0,
                  "system_loss": config.system_loss, "tracking": config.tracking,
                  "tilt": config.tilt, "azim": config.azim, "format": "json"}
        try:
            record_request(api_token)
            response = session.get(NINJA_API_URL, headers=headers, params=params, timeout=60)
            record_response(api_token, response)
        except requests.RequestException as exc:
            raise WeatherDataError(f"Could not discover Renewables.ninja date range: {exc}") from exc
        if response.status_code in {401, 403}:
            raise WeatherDataError(f"Renewables.ninja rejected the API token (HTTP {response.status_code})")
        if response.ok:
            try:
                data = response.json().get("data", {})
            except ValueError:
                data = {}
            if isinstance(data, dict) and len(data) >= 24:
                return year
        time.sleep(1.05)
    raise WeatherDataError("Could not find an available Renewables.ninja year from 2000 onward")


def _request_ninja_year(
    config: WeatherConfig, year: int, token: str, *, session: requests.Session | None = None,
    max_retries: int = 2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = session or requests.Session()
    client.trust_env = False
    headers = {"Authorization": f"Token {token}", "Accept": "application/json",
               "User-Agent": "sota-sabatier/0.1 (research prototype)"}
    params = {
        "lat": config.lat, "lon": config.lon, "date_from": f"{year}-01-01",
        "date_to": f"{year}-12-31", "dataset": config.dataset, "capacity": 1.0,
        "system_loss": config.system_loss, "tracking": config.tracking,
        "tilt": config.tilt, "azim": config.azim, "format": "json",
        "raw": "true", "local_time": "false",
    }
    response: requests.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            record_request(token)
            response = client.get(NINJA_API_URL, headers=headers, params=params,
                                  timeout=NINJA_READ_TIMEOUT_SECONDS)
            record_response(token, response)
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise WeatherDataError(f"Could not reach Renewables.ninja: {exc}") from exc
            time.sleep(2**attempt)
            continue
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
            break
        try:
            time.sleep(min(60.0, float(response.headers.get("Retry-After", 2**attempt))))
        except ValueError:
            time.sleep(2**attempt)
    assert response is not None
    if response.status_code >= 400:
        raise WeatherDataError(f"Renewables.ninja HTTP {response.status_code}: {response.text.strip()[:500]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise WeatherDataError("Renewables.ninja returned invalid JSON") from exc
    raw_data = payload.get("data")
    if not isinstance(raw_data, dict):
        raise WeatherDataError("Renewables.ninja response has no data object")
    rows: list[dict[str, Any]] = []
    for timestamp, values in raw_data.items():
        values = values if isinstance(values, dict) else {"electricity": values}
        temperature = next((values[k] for k in ("temperature", "air_temperature", "t2m") if k in values), np.nan)
        if pd.notna(temperature) and float(temperature) < 170.0:
            temperature = float(temperature) + 273.15
        rows.append({"timestamp": timestamp,
                     "capacity_factor": values.get("electricity", values.get("capacity_factor")),
                     "ambient_temperature_k": temperature})
    frame = pd.DataFrame(rows)
    frame["timestamp"] = _parse_api_timestamps(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    frame["capacity_factor"] = pd.to_numeric(frame["capacity_factor"], errors="coerce")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return validate_hourly_profile(frame, expected_year=year), metadata


def _parse_api_timestamps(values: pd.Series) -> pd.DatetimeIndex:
    """Accept both ISO strings and Renewables.ninja epoch-millisecond keys."""
    text = values.astype(str)
    numeric = text.str.fullmatch(r"\d{12,}")
    if bool(numeric.all()):
        return pd.DatetimeIndex(pd.to_datetime(pd.to_numeric(text), unit="ms", utc=True))
    return pd.DatetimeIndex(pd.to_datetime(text, utc=True))


def validate_hourly_profile(frame: pd.DataFrame, expected_year: int | None = None) -> pd.DataFrame:
    """Validate and normalize an hourly UTC solar profile."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise WeatherDataError("Weather profile must use a DatetimeIndex")
    result = frame.copy()
    # pandas 3 builds datetimes at microsecond resolution where pandas 2 used
    # nanoseconds. Pin one unit at the boundary every profile passes through, so
    # generated, fetched and reloaded records all compare equal on either version.
    result.index = pd.to_datetime(result.index, utc=True).as_unit("ns")
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if "capacity_factor" not in result:
        raise WeatherDataError("Weather profile lacks capacity_factor")
    result["capacity_factor"] = pd.to_numeric(result["capacity_factor"], errors="coerce")
    if result["capacity_factor"].isna().any():
        raise WeatherDataError("Weather profile contains missing capacity factors")
    if ((result["capacity_factor"] < -1e-9) | (result["capacity_factor"] > 1.0 + 1e-9)).any():
        raise WeatherDataError("Capacity factors must lie between zero and one")
    result["capacity_factor"] = result["capacity_factor"].clip(0.0, 1.0)
    if expected_year is not None:
        expected = pd.date_range(f"{expected_year}-01-01", f"{expected_year + 1}-01-01",
                                 freq="h", inclusive="left", tz="UTC")
        missing, extra = expected.difference(result.index), result.index.difference(expected)
        if len(missing) or len(extra):
            raise WeatherDataError(f"Year {expected_year} is incomplete: {len(missing)} missing and {len(extra)} unexpected hours")
        result = result.reindex(expected)
    if "ambient_temperature_k" not in result:
        result["ambient_temperature_k"] = np.nan
    return result[["capacity_factor", "ambient_temperature_k"]]


def fetch_solar_profile(
    config: WeatherConfig, years: Sequence[int] | None = None, *, token: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR, progress: ProgressCallback | None = None,
    allow_api: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load solar data from yearly cache, fetching missing years from Renewables.ninja.

    With ``allow_api=False`` no request is ever issued: any year missing from the
    cache raises instead. That makes it safe to explore settings against already
    downloaded weather without risking metered calls.
    """
    if not (-90 <= config.lat <= 90 and -180 <= config.lon <= 180):
        raise ValueError("Latitude/longitude are outside valid bounds")
    count = config.training_years + config.evaluation_years
    api_token = (token or _load_api_token()) if allow_api else None
    if years is None:
        latest = config.latest_year
        cached = cached_weather_years(config, cache_dir)
        if latest is None and len(cached) >= count:
            for candidate in sorted(cached, reverse=True):
                required = list(range(candidate - count + 1, candidate + 1))
                if set(required).issubset(cached):
                    latest = candidate
                    break
        if latest is None:
            if not allow_api:
                raise WeatherDataError(
                    "Cached-only weather needs an explicit latest year; none of the "
                    "cached years form a complete run for this site."
                )
            latest = discover_latest_weather_year(config, token=api_token)
        years = list(range(latest - count + 1, latest + 1))
    years = sorted(set(int(year) for year in years))
    if years:
        _report_progress(progress, f"Weather: checking {len(years)} yearly files for {years[0]}–{years[-1]}.")
    frames, metadata_by_year = [], {}
    session = requests.Session()
    for year in years:
        csv_path, metadata_path = _cache_paths(config, year, cache_dir)
        if csv_path.exists():
            _report_progress(progress, f"Weather {year}: loading cached data.")
            cached_frame = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")
            frame = validate_hourly_profile(cached_frame, expected_year=year)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        else:
            if not allow_api:
                raise WeatherDataError(
                    f"Year {year} is not in the weather cache for this site and "
                    "cached-only mode is selected, so it will not be downloaded. "
                    "Switch to Renewables.ninja to fetch it."
                )
            if not api_token:
                raise WeatherDataError(f"Year {year} is not cached and RENEWABLES_NINJA_TOKEN is not configured")
            _report_progress(progress, f"Weather {year}: downloading from Renewables.ninja.")
            frame, metadata = _request_ninja_year(config, year, api_token, session=session)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            frame.rename_axis("timestamp").to_csv(csv_path)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            time.sleep(1.05)
        frames.append(frame)
        metadata_by_year[str(year)] = metadata
    _report_progress(progress, "Weather: all requested years validated.")
    combined = pd.concat(frames).sort_index()
    return combined, {"years": years, "training_years": years[:config.training_years],
                      "evaluation_years": years[config.training_years:],
                      "cache_key": _weather_cache_key(config), "source": "Renewables.ninja",
                      "metadata_by_year": metadata_by_year}


def split_weather_period(profile: pd.DataFrame, training_years: int = 5,
                         evaluation_years: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(profile.index.year.unique())
    required = training_years + evaluation_years
    if len(years) < required:
        raise WeatherDataError(f"Need {required} complete years; profile contains {len(years)}")
    selected = years[-required:]
    if selected != list(range(selected[0], selected[-1] + 1)):
        raise WeatherDataError("Training/evaluation years must be contiguous")
    return (profile[profile.index.year.isin(selected[:training_years])],
            profile[profile.index.year.isin(selected[training_years:])])


def build_climatology_forecast(training: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    """Average training values by month/day/hour and align them to evaluation timestamps."""
    train, evaluation = validate_hourly_profile(training), validate_hourly_profile(evaluation)
    keyed = train.assign(month=train.index.month, day=train.index.day, hour=train.index.hour)
    means = keyed.groupby(["month", "day", "hour"])[["capacity_factor", "ambient_temperature_k"]].mean()
    rows = []
    for stamp in evaluation.index:
        key = (stamp.month, stamp.day, stamp.hour)
        if key in means.index:
            values = means.loc[key]
        elif stamp.month == 2 and stamp.day == 29:
            values = (means.loc[(2, 28, stamp.hour)] + means.loc[(3, 1, stamp.hour)]) / 2
        else:
            raise WeatherDataError(f"No climatology value for {stamp.month:02d}-{stamp.day:02d} {stamp.hour:02d}:00")
        rows.append((float(values["capacity_factor"]), float(values["ambient_temperature_k"])))
    result = pd.DataFrame(rows, index=evaluation.index,
                          columns=["capacity_factor", "ambient_temperature_k"])
    result["ambient_temperature_k"] = result["ambient_temperature_k"].fillna(evaluation["ambient_temperature_k"])
    return result


# Plant mass, energy, and thermal calculations ---------------------------------------


def _plant_molar_masses(plant: PlantParameters) -> dict[str, float]:
    return {
        "CH4": plant.methane_molar_mass_kg_per_mol,
        "H2": plant.hydrogen_molar_mass_kg_per_mol,
        "CO2": plant.carbon_dioxide_molar_mass_kg_per_mol,
        "H2O": plant.water_molar_mass_kg_per_mol,
        "CaO": plant.calcium_oxide_molar_mass_kg_per_mol,
        "CaCO3": plant.calcium_carbonate_molar_mass_kg_per_mol,
    }


def calculate_plant_energy(params: PlantParameters = PlantParameters()) -> PlantEnergyResult:
    """Calculate the auditable full-load electrical requirement per kg methane."""
    if not 0 <= params.air_exhaust_hx_effectiveness <= 1:
        raise ValueError("Air/exhaust heat-exchanger effectiveness must lie between zero and one")
    s = calculations.methane_stoichiometry(
        molar_masses_kg_per_mol=_plant_molar_masses(params)
    )
    calcination_equilibrium_pressure = calculations.calcination_equilibrium_pressure_bar(
        params.calciner_temperature_k,
        params.calcination_delta_h_j_per_mol,
        params.calcination_delta_s_j_per_mol_k,
        params.gas_constant_j_per_mol_k,
    )
    # Carbonator feasibility. The plan requires a reported margin, so evaluate the
    # same van't Hoff relation at the carbonator and compare it against the CO2
    # partial pressure the air actually offers.
    carbonation_equilibrium_pressure = calculations.calcination_equilibrium_pressure_bar(
        params.carbonator_temperature_k,
        params.calcination_delta_h_j_per_mol,
        params.calcination_delta_s_j_per_mol_k,
        params.gas_constant_j_per_mol_k,
    )
    air_co2_partial_pressure = params.air_co2_mole_fraction * params.air_pressure_bar
    carbonation_driving_force_ratio = (
        air_co2_partial_pressure / carbonation_equilibrium_pressure
        if carbonation_equilibrium_pressure > 0 else math.inf
    )
    maximum_carbonator_temperature = calculations.maximum_carbonation_temperature_k(
        air_co2_partial_pressure,
        params.calcination_delta_h_j_per_mol,
        params.calcination_delta_s_j_per_mol_k,
        params.gas_constant_j_per_mol_k,
    )
    carbonation_warning = None
    if carbonation_driving_force_ratio <= 1.0:
        carbonation_warning = (
            f"Carbonator at {params.carbonator_temperature_k:,.2f} K cannot capture CO2 from "
            f"{params.air_co2_mole_fraction * 1e6:,.0f} ppm air: the CaO/CaCO3 equilibrium "
            f"pressure is {carbonation_equilibrium_pressure:.4g} bar against an offered "
            f"{air_co2_partial_pressure:.4g} bar, so CaCO3 decomposes instead of forming. "
            f"Operate below {maximum_carbonator_temperature:,.2f} K."
        )
    compressor_per_co2, compressor_warning = calculations.co2_compression_work_kwh_per_kg(
        calcination_equilibrium_pressure,
        params.co2_outlet_pressure_bar,
        params.intercool_temperature_k,
        params.compressor_isentropic_efficiency,
        params.compressor_stages,
        params.co2_compression_fallback_cp_j_per_kg_k,
        params.co2_compression_fallback_gamma,
    )
    storage_work: dict[str, float] = {}
    storage_warnings: list[str] = []
    gas_specs = {
        "hydrogen": ("Hydrogen", params.co2_outlet_pressure_bar,
                     params.hydrogen_storage_pressure_bar,
                     params.h2_sensible_fallback_cp_j_per_kg_k,
                     params.hydrogen_compression_fallback_gamma),
        "co2": ("CO2", params.co2_outlet_pressure_bar,
                params.co2_storage_pressure_bar,
                params.co2_compression_fallback_cp_j_per_kg_k,
                params.co2_compression_fallback_gamma),
        "methane": ("Methane", params.co2_outlet_pressure_bar,
                    params.methane_storage_pressure_bar,
                    params.methane_compression_fallback_cp_j_per_kg_k,
                    params.methane_compression_fallback_gamma),
    }
    for name, (fluid, connection_pressure, storage_pressure, cp, gamma) in gas_specs.items():
        compression, warning = calculations.gas_compression_work_kwh_per_kg(
            fluid, connection_pressure, storage_pressure, params.intercool_temperature_k,
            params.compressor_isentropic_efficiency, params.storage_machine_stages, cp, gamma,
        )
        expansion_outlet = (
            params.methane_delivery_pressure_bar if name == "methane" else connection_pressure
        )
        expansion, expansion_reheat, expansion_warning = calculations.gas_expansion_work_kwh_per_kg(
            fluid, storage_pressure, expansion_outlet, params.intercool_temperature_k,
            params.storage_expander_isentropic_efficiency, params.storage_machine_stages, cp, gamma,
        )
        storage_work[f"{name}_compression"] = compression
        storage_work[f"{name}_expansion"] = expansion
        storage_work[f"{name}_expansion_reheat"] = expansion_reheat
        storage_warnings.extend(item for item in (warning, expansion_warning) if item)
    electrolysis = calculations.electrolysis_duty_kwh(
        s["H2"], params.electrolyser_kwh_per_kg_h2
    )
    fan = calculations.fan_duty_kwh(s["CO2"], params.fan_kwh_per_kg_co2)
    compression = s["CO2"] * compressor_per_co2
    # Calcination heat comes from the same enthalpy that fixes the CaO/CaCO3
    # equilibrium pressure, rather than a second hand-entered kWh/kg constant
    # that could be edited out of step with it.
    calcination_kwh_per_kg_co2 = (
        params.calcination_delta_h_j_per_mol
        / params.carbon_dioxide_molar_mass_kg_per_mol / 3.6e6
    )
    calcination = calculations.reaction_heat_kwh(s["CO2"], calcination_kwh_per_kg_co2)
    solids = calculations.solids_sensible_loss_kwh(
        s["CaO_circulating"], s["CaCO3_circulating"],
        params.cao_cp_kwh_per_kg_k, params.caco3_cp_kwh_per_kg_k,
        params.carbonator_temperature_k, params.calciner_temperature_k,
        params.solids_heat_recovery_efficiency,
    )
    co2_feed_heat, co2_feed_warning = calculations.gas_sensible_heat_kwh_per_kg(
        "CO2", params.intercool_temperature_k, params.sabatier_temperature_k,
        params.co2_outlet_pressure_bar,
        params.co2_sensible_fallback_cp_j_per_kg_k,
    )
    h2_feed_heat, h2_feed_warning = calculations.gas_sensible_heat_kwh_per_kg(
        "H2", params.intercool_temperature_k, params.sabatier_temperature_k,
        params.co2_outlet_pressure_bar,
        params.h2_sensible_fallback_cp_j_per_kg_k,
    )
    gross_sabatier_feed_heat = s["CO2"] * co2_feed_heat + s["H2"] * h2_feed_heat
    sabatier_heat_to_feed = min(gross_sabatier_feed_heat, params.sabatier_heat_kwh_per_kg_ch4)
    electric_sabatier_feed_heat = max(0.0, gross_sabatier_feed_heat - sabatier_heat_to_feed)
    air_mass = calculations.dry_air_mass_for_co2_kg(
        s["CO2"], params.air_co2_mole_fraction, params.air_molar_mass_kg_per_mol,
        params.carbon_dioxide_molar_mass_kg_per_mol,
        params.dac_capture_efficiency,
    )
    air_specific_heat, air_warning = calculations.air_sensible_heat_kwh_per_kg(
        params.reference_ambient_temperature_k, params.carbonator_temperature_k,
        params.air_pressure_bar, params.air_fallback_cp_kwh_per_kg_k,
    )
    gross_air_heat = air_mass * air_specific_heat
    residual_air_heat = gross_air_heat * (1.0 - params.air_exhaust_hx_effectiveness)
    breakdown = {
        "electrolysis": electrolysis, "dac_fan": fan, "co2_compression": compression,
        "calcination_reaction_heat": calcination,
        "unrecovered_solids_sensible_heat": solids["unrecovered_solids_sensible_heat"],
        "sabatier_feed_electric_heat": electric_sabatier_feed_heat,
        "carbonator_air_electric_heat": residual_air_heat,
        "methane_storage_compression": storage_work["methane_compression"],
        "methane_storage_expansion_recovered": -storage_work["methane_expansion"],
        # Interstage reheat keeping the expander out of the two-phase region, supplied
        # by electrical heating at COP 1 like every other heat duty in the model.
        "methane_storage_expansion_reheat": storage_work["methane_expansion_reheat"],
        "sabatier_heat_recovered_to_feed": sabatier_heat_to_feed,
        "sabatier_heat_discarded": max(
            0.0, params.sabatier_heat_kwh_per_kg_ch4 - sabatier_heat_to_feed
        ),
    }
    e_req = calculations.sum_specific_electricity_kwh_per_kg(
        breakdown,
        excluded_keys=("sabatier_heat_recovered_to_feed", "sabatier_heat_discarded"),
    )
    return PlantEnergyResult(
        e_req, breakdown, s,
        calcination_equilibrium_pressure,
        compressor_per_co2,
        tuple(x for x in [carbonation_warning, compressor_warning, co2_feed_warning,
                          h2_feed_warning, air_warning, *storage_warnings] if x),
        storage_work,
        carbonation_equilibrium_pressure_bar=carbonation_equilibrium_pressure,
        air_co2_partial_pressure_bar=air_co2_partial_pressure,
        carbonation_driving_force_ratio=carbonation_driving_force_ratio,
        maximum_carbonator_temperature_k=maximum_carbonator_temperature,
    )


def _thermal_inventory_and_capacity(nominal: float, plant: PlantParameters) -> tuple[float, float, float, float]:
    s = calculations.methane_stoichiometry(
        molar_masses_kg_per_mol=_plant_molar_masses(plant)
    )
    cao = s["CaO_circulating"] * nominal * plant.solids_cycle_time_h
    caco3 = s["CaCO3_circulating"] * nominal * plant.solids_cycle_time_h
    return cao, caco3, max(1e-9, cao * plant.cao_cp_kwh_per_kg_k), max(1e-9, caco3 * plant.caco3_cp_kwh_per_kg_k)


def _ambient_series(frame: pd.DataFrame, fallback_k: float) -> pd.Series:
    if "ambient_temperature_k" not in frame:
        return pd.Series(fallback_k, index=frame.index)
    return pd.to_numeric(frame["ambient_temperature_k"], errors="coerce").fillna(fallback_k)


def nominal_methane_rate(profile: pd.DataFrame, energy: PlantEnergyResult, plant: PlantParameters,
                         strategy: StrategyConfig) -> float:
    cf = profile["capacity_factor"]
    if strategy.short_strategy == "through_night":
        cf_long = float(cf.mean())
    else:
        daylight = cf > strategy.daylight_cf_cutoff
        cf_long = float(cf[daylight].mean()) if daylight.any() else 0.0
    return 1000.0 * plant.solar_farm_mw * cf_long / (energy.e_req_kwh_per_kg_ch4 * (1 + strategy.f_ocp))


def round_reactor_equivalents(equivalents: float, reactor_count: int) -> int:
    """Round a daily continuous train requirement half-up to an installed count."""
    if reactor_count < 1:
        raise ValueError("reactor_count must be positive")
    if not math.isfinite(equivalents):
        raise ValueError("equivalents must be finite")
    return min(reactor_count, max(0, math.floor(equivalents + 0.5)))


def _common_thermal_schedule(
        forecast: pd.DataFrame, nominal: float, plant: PlantParameters,
        thermal: ThermalParameters, strategy: StrategyConfig,
        operating: pd.Series) -> pd.DataFrame:
    """Return common carbonator/calciner loads for a proposed operating mask."""
    index = forecast.index
    _, _, carb_c, calc_c = _thermal_inventory_and_capacity(nominal, plant)
    ambient = _ambient_series(forecast, thermal.ambient_fallback_k)
    t_carb, t_calc = plant.carbonator_temperature_k, plant.calciner_temperature_k
    standby = np.zeros(len(index), dtype=float)
    startup = np.zeros(len(index), dtype=float)
    states = np.empty(len(index), dtype=object)
    previous_operating = False
    for pos in range(len(index)):
        amb = float(ambient.iloc[pos])
        is_operating = bool(operating.iloc[pos])
        if strategy.short_strategy == "hard_shutdown" and not is_operating:
            t_carb = amb + (t_carb - amb) * math.exp(-thermal.carbonator_ua_kw_per_k / carb_c)
            t_calc = amb + (t_calc - amb) * math.exp(-thermal.calciner_ua_kw_per_k / calc_c)
            states[pos] = "cooling"
        else:
            if strategy.short_strategy == "hard_shutdown" and is_operating and not previous_operating:
                startup[pos] = (
                    max(0.0, carb_c * (plant.carbonator_temperature_k - t_carb))
                    + max(0.0, calc_c * (plant.calciner_temperature_k - t_calc))
                )
                t_carb, t_calc = plant.carbonator_temperature_k, plant.calciner_temperature_k
            standby[pos] = (
                thermal.carbonator_ua_kw_per_k * max(0.0, t_carb - amb)
                + thermal.calciner_ua_kw_per_k * max(0.0, t_calc - amb)
                + thermal.fixed_hot_auxiliary_kw
            )
            states[pos] = "operating" if is_operating else "limping"
        previous_operating = is_operating
    return pd.DataFrame(
        {"common_thermal_load_kw": standby, "common_startup_load_kwh": startup,
         "planned_state": states}, index=index,
    )


def sabatier_train_thermal_properties(
        nominal: float, thermal: ThermalParameters,
        reactor_count: int) -> tuple[float, float]:
    """Return per-train thermal capacity (kWh/K) and UA (kW/K)."""
    unit_capacity = max(0.0, nominal) / reactor_count
    scale = unit_capacity / thermal.sabatier_reference_capacity_kg_ch4_h
    heat_capacity = thermal.sabatier_reference_thermal_capacity_kwh_per_k * scale
    ua = thermal.sabatier_reference_ua_kw_per_k * scale ** thermal.sabatier_ua_scaling_exponent
    return max(1e-12, heat_capacity), max(1e-12, ua)


def _planned_sabatier_thermal_schedule(
        forecast: pd.DataFrame, nominal: float, plant: PlantParameters,
        thermal: ThermalParameters, strategy: StrategyConfig,
        active_count: pd.Series) -> pd.DataFrame:
    """Track planned per-train standby heat startup heat and natural cooling."""
    count = strategy.parallel_reactor_count
    heat_capacity, ua = sabatier_train_thermal_properties(nominal, thermal, count)
    temperatures = np.full(count, plant.sabatier_temperature_k, dtype=float)
    ambient = _ambient_series(forecast, thermal.ambient_fallback_k)
    standby = np.zeros(len(forecast), dtype=float)
    startup = np.zeros(len(forecast), dtype=float)
    mean_temperature = np.zeros(len(forecast), dtype=float)
    minimum_temperature = np.zeros(len(forecast), dtype=float)
    for pos in range(len(forecast)):
        amb = float(ambient.iloc[pos])
        active = int(active_count.iloc[pos])
        order = np.argsort(-temperatures)
        active_ids = order[:active]
        inactive_ids = order[active:]
        if strategy.short_strategy == "hard_shutdown":
            if active:
                startup[pos] = float(np.maximum(
                    0.0, heat_capacity * (plant.sabatier_temperature_k - temperatures[active_ids])
                ).sum())
                temperatures[active_ids] = plant.sabatier_temperature_k
            if len(inactive_ids):
                decay = math.exp(-ua / heat_capacity)
                temperatures[inactive_ids] = amb + (temperatures[inactive_ids] - amb) * decay
        else:
            if len(inactive_ids):
                standby[pos] = float(
                    ua * np.maximum(0.0, plant.sabatier_temperature_k - amb) * len(inactive_ids)
                )
                temperatures[inactive_ids] = plant.sabatier_temperature_k
            if active:
                temperatures[active_ids] = plant.sabatier_temperature_k
        mean_temperature[pos] = float(temperatures.mean())
        minimum_temperature[pos] = float(temperatures.min())
    return pd.DataFrame(
        {"sabatier_thermal_load_kw": standby, "sabatier_startup_load_kwh": startup,
         "planned_sabatier_mean_temperature_k": mean_temperature,
         "planned_sabatier_min_temperature_k": minimum_temperature},
        index=forecast.index,
    )


def build_target_schedule(forecast: pd.DataFrame, nominal: float, energy: PlantEnergyResult,
                          plant: PlantParameters, thermal: ThermalParameters,
                          strategy: StrategyConfig) -> pd.DataFrame:
    """Build integer-train targets from daily energy or seasonal CF averages."""
    forecast = validate_hourly_profile(forecast)
    index = forecast.index
    count = strategy.parallel_reactor_count
    unit_rate = nominal / count
    daylight = forecast["capacity_factor"] > strategy.daylight_cf_cutoff
    provisional_operating = pd.Series(True, index=index) if strategy.short_strategy == "through_night" else daylight
    if strategy.reactor_scheduling_mode != "seasonal":
        common = _common_thermal_schedule(
            forecast, nominal, plant, thermal, strategy, provisional_operating,
        )
        sabatier = pd.DataFrame({
            "sabatier_thermal_load_kw": np.zeros(len(index)),
            "sabatier_startup_load_kwh": np.zeros(len(index)),
        }, index=index)
    generation = forecast["capacity_factor"] * plant.solar_farm_mw * 1000.0
    continuous = pd.Series(0.0, index=index)
    daily_count = pd.Series(0, index=index, dtype=int)
    day_groups = pd.Series(index.normalize(), index=index).groupby(index.normalize()).groups
    seasonal_continuous: dict[int, float] = {}
    seasonal_counts: dict[int, int] = {}
    if strategy.reactor_scheduling_mode == "seasonal":
        # Seasonal means use the CF values available during the selected
        # operating hours.  Summer is the reference season and is assigned
        # the full train count; the other seasons are scaled relative to it.
        season_for_month = {
            12: 0, 1: 0, 2: 0,       # winter
            3: 1, 4: 1, 5: 1,        # spring
            6: 2, 7: 2, 8: 2,        # summer
            9: 3, 10: 3, 11: 3,      # autumn
        }
        season_labels = index.month.map(season_for_month)
        operating_mask = (
            pd.Series(True, index=index)
            if strategy.short_strategy == "through_night" else daylight
        )
        seasonal_means = forecast.loc[operating_mask, "capacity_factor"].groupby(
            season_labels[operating_mask]
        ).mean()
        summer_cf = float(seasonal_means.get(2, 0.0))
        for season_number in range(4):
            season_cf = float(seasonal_means.get(season_number, 0.0))
            continuous_value = (
                0.0 if summer_cf <= 0.0 else min(float(count), max(0.0, count * season_cf / summer_cf))
            )
            seasonal_continuous[season_number] = continuous_value
            seasonal_counts[season_number] = max(
                1, round_reactor_equivalents(continuous_value, count)
            )
    if strategy.reactor_scheduling_mode == "seasonal":
        # Seasonal commitments depend only on the four forecast CF averages. Avoid
        # the daily storage-aware schedule's convergence loop: its daily energy
        # equivalents cannot change these fixed seasonal counts.
        season_codes = np.fromiter(
            (season_for_month[int(month)] for month in index.month),
            dtype=np.int8,
            count=len(index),
        )
        continuous = pd.Series(
            np.asarray([seasonal_continuous[number] for number in range(4)])[season_codes],
            index=index,
        )
        daily_count = pd.Series(
            np.asarray([seasonal_counts[number] for number in range(4)], dtype=int)[season_codes],
            index=index,
            dtype=int,
        )
        active_count = daily_count.copy()
        if strategy.short_strategy != "through_night":
            active_count = active_count.where(daylight, 0)
        common = _common_thermal_schedule(
            forecast, nominal, plant, thermal, strategy, active_count > 0,
        )
        sabatier = _planned_sabatier_thermal_schedule(
            forecast, nominal, plant, thermal, strategy, active_count,
        )
    else:
        previous_daily_count: pd.Series | None = None
        # Thermal overhead depends on the rounded count. Iterate the deterministic
        # schedule to include common and train standby/startup energy in the daily budget.
        for _ in range(8):
            overhead = (
                common["common_thermal_load_kw"] + common["common_startup_load_kwh"]
                + sabatier["sabatier_thermal_load_kw"] + sabatier["sabatier_startup_load_kwh"]
            )
            for positions in day_groups.values():
                day_index = pd.DatetimeIndex(positions)
                if strategy.short_strategy == "through_night":
                    operating_hours = len(day_index)
                else:
                    operating_hours = int(daylight.loc[day_index].sum())
                unit_process_energy = unit_rate * energy.e_req_kwh_per_kg_ch4 * operating_hours
                available = float(generation.loc[day_index].sum() - overhead.loc[day_index].sum())
                equivalents = (
                    0.0 if unit_process_energy <= 0
                    else max(0.0, available) / unit_process_energy
                )
                equivalents = min(float(count), equivalents)
                continuous.loc[day_index] = equivalents
                daily_count.loc[day_index] = round_reactor_equivalents(equivalents, count)
            active_count = daily_count.copy()
            if strategy.short_strategy != "through_night":
                active_count = active_count.where(daylight, 0)
            common = _common_thermal_schedule(
                forecast, nominal, plant, thermal, strategy, active_count > 0,
            )
            sabatier = _planned_sabatier_thermal_schedule(
                forecast, nominal, plant, thermal, strategy, active_count,
            )
            if previous_daily_count is not None and daily_count.equals(previous_daily_count):
                break
            previous_daily_count = daily_count.copy()
    methane = active_count.astype(float) * unit_rate
    process = methane * energy.e_req_kwh_per_kg_ch4
    thermal_load = common["common_thermal_load_kw"] + sabatier["sabatier_thermal_load_kw"]
    startup_load = common["common_startup_load_kwh"] + sabatier["sabatier_startup_load_kwh"]
    result = pd.DataFrame({
        "forecast_cf": forecast["capacity_factor"],
        "continuous_reactor_equivalents": continuous,
        "daily_reactor_count": daily_count,
        "planned_reactor_count": active_count,
        "target_methane_kg_h": methane,
        "process_load_kw": process,
        "thermal_load_kw": thermal_load,
        "startup_load_kwh": startup_load,
        "target_total_load_kw": process + thermal_load + startup_load,
        "planned_state": common["planned_state"],
        "daylight": daylight,
    }, index=index)
    return result.join(common.drop(columns="planned_state")).join(sabatier)


# Storage sizing and dispatch ---------------------------------------------------------


def _carrier_kwh_per_charge_kwh(store: StorageParameters) -> float:
    if store.method == "battery":
        return 1.0
    if store.method == "h2_co2":
        input_per_kg_h2 = (
            store.hydrogen_electrolyser_kwh_per_kg
            + store.hydrogen_compression_kwh_per_kg
            + store.co2_to_hydrogen_mass_ratio * store.co2_production_kwh_per_kg
            + store.co2_to_hydrogen_mass_ratio * store.co2_compression_kwh_per_kg
        )
        return store.hydrogen_lhv_kwh_per_kg / input_per_kg_h2
    return store.hydrogen_lhv_kwh_per_kg / (
        store.hydrogen_electrolyser_kwh_per_kg + store.hydrogen_compression_kwh_per_kg
    )


def _net_h2_bus_kwh_per_kg(store: StorageParameters) -> float:
    """Bus electricity per kg of stored H2 sent through the expander and fuel cell.

    The expander's interstage reheat is an electrical heating load at COP 1, so it
    is deducted from the recovered expansion work.
    """
    return (store.hydrogen_fuel_cell_kwh_per_kg
            + store.hydrogen_expansion_kwh_per_kg
            - store.hydrogen_expansion_reheat_kwh_per_kg)


def _discharge_kwh_per_carrier_kwh(store: StorageParameters) -> float:
    if store.method == "battery":
        return 1.0
    return _net_h2_bus_kwh_per_kg(store) / store.hydrogen_lhv_kwh_per_kg


def _storage_increment_from_bus(flow: np.ndarray, store: StorageParameters) -> np.ndarray:
    charge_ratio = _carrier_kwh_per_charge_kwh(store)
    discharge_ratio = _discharge_kwh_per_carrier_kwh(store)
    return np.where(flow >= 0, flow * charge_ratio, flow / discharge_ratio)


def _cyclic_bus_energy_deficit(flow: np.ndarray, store: StorageParameters) -> float:
    """Return extra bus-side charge energy needed to make a cycle periodic."""
    flow = np.asarray(flow, dtype=float)
    if len(flow) == 0:
        return 0.0
    charge_ratio = _carrier_kwh_per_charge_kwh(store)
    discharge_ratio = _discharge_kwh_per_carrier_kwh(store)
    positive_bus = np.maximum(flow, 0.0)
    negative_internal = np.minimum(flow, 0.0) / discharge_ratio
    if store.self_discharge_fraction_per_h <= 0:
        internal_deficit = max(
            0.0,
            -float((positive_bus * charge_ratio + negative_internal).sum()),
        )
        return internal_deficit / charge_ratio

    retention = 1.0 - store.self_discharge_fraction_per_h

    def minimum_periodic_soc(charge_multiplier: float) -> float:
        increments = (
            positive_bus * charge_ratio * charge_multiplier + negative_internal
        )
        forcing = 0.0
        for increment in increments:
            forcing = retention * forcing + increment
        denominator = 1.0 - retention ** len(increments)
        initial = forcing / denominator if denominator > 1e-15 else 0.0
        minimum = initial
        soc = initial
        for increment in increments:
            soc = retention * soc + increment
            minimum = min(minimum, soc)
        return minimum

    if minimum_periodic_soc(1.0) >= -1e-7:
        return 0.0
    positive_total = float(positive_bus.sum())
    if positive_total <= 1e-12:
        # No existing charging window is available. Report the loss-free bus energy
        # shortfall; the case remains non-cyclic and dispatch records the consequences.
        return max(0.0, -float(negative_internal.sum()) / charge_ratio)
    low, high = 1.0, 2.0
    while minimum_periodic_soc(high) < -1e-7 and high < 1e9:
        high *= 2.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if minimum_periodic_soc(middle) >= -1e-7:
            high = middle
        else:
            low = middle
    return positive_total * (high - 1.0)


def _cyclic_capacity(flow: np.ndarray, store: StorageParameters) -> tuple[float, float, float, float, bool, str | None]:
    if len(flow) == 0:
        return 0.0, 0.0, 0.0, 0.0, True, None
    if store.self_discharge_fraction_per_h > 0:
        return _cyclic_capacity_with_self_discharge(np.asarray(flow, dtype=float), store)
    increments = _storage_increment_from_bus(np.asarray(flow, dtype=float), store)
    total = float(increments.sum())
    feasible, warning = total >= -1e-6, None
    if total > 1e-9:
        positive, positive_total = increments > 0, float(increments[increments > 0].sum())
        if positive_total:
            increments[positive] *= max(0.0, 1.0 - total / positive_total)
    elif total < -1e-6:
        warning = f"Cyclic storage balance has an energy deficit of {-total:.3f} kWh."
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    minimum, maximum = float(cumulative.min()), float(cumulative.max())
    return maximum - minimum, -minimum, float(np.maximum(flow, 0).max(initial=0.0)), \
        float(np.maximum(-flow, 0).max(initial=0.0)), feasible, warning


def _cyclic_capacity_with_self_discharge(
    flow: np.ndarray, store: StorageParameters
) -> tuple[float, float, float, float, bool, str | None]:
    """Solve the unique periodic SOC trajectory including hourly self-discharge."""
    retention = 1.0 - store.self_discharge_fraction_per_h
    positive_increment = np.maximum(flow, 0) * _carrier_kwh_per_charge_kwh(store)
    negative_increment = np.minimum(flow, 0) / _discharge_kwh_per_carrier_kwh(store)

    def trajectory(charge_fraction: float) -> tuple[float, np.ndarray]:
        increments = positive_increment * charge_fraction + negative_increment
        forcing = 0.0
        for increment in increments:
            forcing = retention * forcing + increment
        denominator = 1.0 - retention ** len(increments)
        initial = forcing / denominator if denominator > 1e-15 else 0.0
        values = np.empty(len(increments) + 1)
        values[0] = initial
        for index, increment in enumerate(increments):
            values[index + 1] = retention * values[index] + increment
        return initial, values

    initial, values = trajectory(1.0)
    if float(values.min()) < -1e-6:
        deficit = -float(values.min())
        return (float(values.max() - values.min()), max(0.0, initial),
                float(np.maximum(flow, 0).max(initial=0.0)),
                float(np.maximum(-flow, 0).max(initial=0.0)), False,
                f"Cyclic storage balance including self-discharge is short by {deficit:.3f} kWh.")
    low, high = 0.0, 1.0
    for _ in range(36):
        middle = (low + high) / 2
        _, candidate = trajectory(middle)
        if float(candidate.min()) >= -1e-8:
            high = middle
        else:
            low = middle
    charge_fraction = high
    initial, values = trajectory(charge_fraction)
    values[np.abs(values) < 1e-7] = 0.0
    return (float(values.max()), max(0.0, initial),
            float((np.maximum(flow, 0) * charge_fraction).max(initial=0.0)),
            float(np.maximum(-flow, 0).max(initial=0.0)), True, None)


def _short_cycle_and_long_residual(
    net: pd.Series, store: StorageParameters
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float, bool, str | None]]:
    """Close each short-store day and pass its losses to the long balance."""
    short_bus = np.zeros(len(net), dtype=float)
    long_bus = np.zeros(len(net), dtype=float)
    capacities: list[float] = []
    initials: list[float] = []
    max_charge = max_discharge = 0.0
    feasible = True
    warnings: list[str] = []
    positions = pd.Series(np.arange(len(net)), index=net.index)
    for _, day_positions in positions.groupby(net.index.floor("D")):
        locs = day_positions.to_numpy(dtype=int)
        values = net.iloc[locs].to_numpy(dtype=float)
        residual = values - float(values.mean())
        increments = _storage_increment_from_bus(residual, store)
        internal_deficit = max(0.0, -float(increments.sum()))
        extra_bus_charge = internal_deficit / _carrier_kwh_per_charge_kwh(store)
        positive = residual > 0
        if extra_bus_charge > 0 and positive.any():
            residual[positive] += extra_bus_charge * residual[positive] / residual[positive].sum()
        elif extra_bus_charge > 0:
            residual += extra_bus_charge / len(residual)
        short_bus[locs] = residual
        long_bus[locs] = values - residual
        day_size = _cyclic_capacity(residual, store)
        capacities.append(day_size[0])
        initials.append(day_size[1])
        max_charge = max(max_charge, day_size[2])
        max_discharge = max(max_discharge, day_size[3])
        feasible = feasible and day_size[4]
        if day_size[5]:
            warnings.append(day_size[5])
    summary = (max(capacities, default=0.0), max(initials, default=0.0),
               max_charge, max_discharge, feasible,
               warnings[0] if warnings else None)
    return short_bus, long_bus, summary


def realised_cyclic_deficit(actual: pd.DataFrame, nominal: float,
                            energy: PlantEnergyResult, plant: PlantParameters,
                            thermal: ThermalParameters, strategy: StrategyConfig,
                            short_storage: StorageParameters,
                            long_storage: StorageParameters) -> float:
    """Annual cyclic energy deficit a already-built plant runs on weather it later meets.

    size_perfect_storage reports the deficit of a plant against the very record it was
    sized from, which is zero by construction whenever the sizing succeeded. A plant
    sized on an earlier, shorter record has no such guarantee: five years contains
    fewer hard winters than ten, so the store it asked for can be too small for the
    decade that follows. This measures that on the weather actually experienced,
    holding throughput and installed capacity at what was built.
    """
    schedule = build_target_schedule(actual, nominal, energy, plant, thermal, strategy)
    net = actual["capacity_factor"] * plant.solar_farm_mw * 1000.0 - schedule["target_total_load_kw"]
    _, long_bus, _ = _short_cycle_and_long_residual(net, short_storage)
    deficit = _cyclic_bus_energy_deficit(long_bus, long_storage)
    years = len(actual) / 8760.0
    return deficit / years if years else 0.0


def size_perfect_storage(actual: pd.DataFrame, energy: PlantEnergyResult,
                         plant: PlantParameters = PlantParameters(),
                         thermal: ThermalParameters = ThermalParameters(),
                         short_storage: StorageParameters = StorageParameters(),
                         long_storage: StorageParameters = StorageParameters(self_discharge_fraction_per_h=1e-5),
                         strategy: StrategyConfig = StrategyConfig(),
                         progress: ProgressCallback | None = None) -> StorageSizingResult:
    """Size storage energy cyclically; run_case measures transfer power after dispatch."""
    if strategy.f_socp_long < 0:
        raise ValueError("Long-term storage energy-capacity factor f_SOCP cannot be negative")
    actual = validate_hourly_profile(actual)
    _report_progress(progress, "Storage energy: constructing the perfect-information SOC trajectory.")
    nominal = nominal_methane_rate(actual, energy, plant, strategy)
    schedule = build_target_schedule(actual, nominal, energy, plant, thermal, strategy)
    net = actual["capacity_factor"] * plant.solar_farm_mw * 1000.0 - schedule["target_total_load_kw"]
    _, long_bus, short = _short_cycle_and_long_residual(net, short_storage)
    long = _cyclic_capacity(long_bus, long_storage)
    balance_deficit = _cyclic_bus_energy_deficit(long_bus, long_storage)
    simulated_years = len(actual) / 8760.0
    annual_balance_deficit = balance_deficit / simulated_years if simulated_years else 0.0
    warnings = list(energy.warnings) + ([short[5]] if short[5] else [])
    if long[5] and balance_deficit <= 1e-6:
        warnings.append(long[5])
    if annual_balance_deficit > 1e-6:
        warnings.append(
            "Selected f_OCP leaves an average cyclic energy deficit of "
            f"{annual_balance_deficit:.3f} kWh/year. Dispatch will continue and report "
            "the resulting production shortfall."
        )
    if nominal <= 0:
        warnings.append("No methane capacity can be sized because the solar profile has no daylight output.")
    if (strategy.reactor_scheduling_mode == "seasonal"
            and strategy.parallel_reactor_count < 2):
        warnings.append(
            "Seasonal scheduling has no effect with a single reactor train; there is no "
            "second train to stand down in winter. Raise the train count to use it."
        )
    short_capacity = short[0]
    long_capacity = long[0]
    long_co2_capacity = (
        long_capacity / long_storage.hydrogen_lhv_kwh_per_kg
        * long_storage.co2_to_hydrogen_mass_ratio
        if long_storage.method == "h2_co2" else 0.0
    )
    # Capacity still comes from the cyclic trajectory range, but the simulation's
    # starting inventory is an explicit user choice shared by every storage carrier.
    short_initial = short_capacity * short_storage.initial_soc_fraction
    long_initial = long_capacity * long_storage.initial_soc_fraction
    long_initial_co2 = long_co2_capacity * long_storage.initial_soc_fraction

    _report_progress(
        progress,
        "Storage energy: cyclic capacities calculated; transfer power will be measured "
        "after the unconstrained dispatch.",
    )

    return StorageSizingResult(
        short_capacity,
        long_capacity,
        0.0,
        0.0,
        0.0,
        0.0,
        short_initial,
        long_initial,
        nominal,
        balance_deficit,
        annual_balance_deficit,
        short[4] and long[4] and balance_deficit <= 1e-6 and nominal > 0,
        tuple(warnings),
        long_co2_capacity,
        long_initial_co2,
    )


def _fault_fraction_at(timestamp: pd.Timestamp, target: str,
                       faults: Sequence[FaultEvent]) -> float:
    timestamp = _utc_timestamp(timestamp)
    capacity = 1.0
    for event in faults:
        if event.component != target:
            continue
        if event.start <= timestamp < event.end:
            capacity = min(capacity, event.capacity_fraction)
    return capacity


def _fault_active_at(timestamp: pd.Timestamp, target: str,
                     faults: Sequence[FaultEvent]) -> bool:
    timestamp = _utc_timestamp(timestamp)
    return any(
        event.component == target and event.start <= timestamp < event.end
        for event in faults
    )


class _StoreState:
    def __init__(self, parameters: StorageParameters, initial_soc: float | None = None,
                 initial_co2_soc_kg: float | None = None):
        self.p = parameters
        self.soc = min(parameters.capacity_kwh, max(0.0, parameters.capacity_kwh * parameters.initial_soc_fraction
                                                    if initial_soc is None else initial_soc))
        self.co2_soc_kg = min(
            parameters.co2_capacity_kg,
            max(0.0, parameters.co2_capacity_kg * parameters.initial_soc_fraction
                if initial_co2_soc_kg is None else initial_co2_soc_kg),
        )
        self.inaccessible_soc = 0.0
        self.max_inaccessible_soc = 0.0
        self.capacity_fraction = 1.0

    def begin_hour(self, capacity_fraction: float) -> None:
        retention = max(0.0, 1 - self.p.self_discharge_fraction_per_h)
        self.soc *= retention
        self.inaccessible_soc *= retention
        self.co2_soc_kg *= max(0.0, 1 - self.p.co2_self_discharge_fraction_per_h)
        capacity_fraction = max(0.0, min(1.0, capacity_fraction))
        available_capacity = self.p.capacity_kwh * capacity_fraction
        # Enforce the accessible-energy ceiling every hour. Restoration only
        # occurs when capacity recovers; discharge does not make faulted cells
        # available early.
        if self.soc > available_capacity:
            newly_inaccessible = self.soc - available_capacity
            self.inaccessible_soc += newly_inaccessible
            self.soc = available_capacity
        elif capacity_fraction > self.capacity_fraction and self.inaccessible_soc > 0:
            restored = min(self.inaccessible_soc, max(0.0, available_capacity - self.soc))
            self.inaccessible_soc -= restored
            self.soc += restored
        self.capacity_fraction = capacity_fraction
        self.max_inaccessible_soc = max(self.max_inaccessible_soc, self.inaccessible_soc)

    def charge(self, bus_energy: float, *,
               max_h2_charge_kg: float = math.inf,
               max_co2_charge_kg: float = math.inf) -> float:
        if bus_energy <= 0 or self.p.capacity_kwh <= 0:
            return 0.0
        if self.p.method == "h2_co2":
            limit = bus_energy
            h2_room_kg = max(
                0.0,
                (self.p.capacity_kwh * self.capacity_fraction - self.soc)
                / self.p.hydrogen_lhv_kwh_per_kg,
            )
            co2_room_kg = max(
                0.0,
                self.p.co2_capacity_kg * self.capacity_fraction - self.co2_soc_kg,
            )
            paired_h2_room_kg = min(
                h2_room_kg,
                co2_room_kg / self.p.co2_to_hydrogen_mass_ratio,
                max_h2_charge_kg,
                max_co2_charge_kg / self.p.co2_to_hydrogen_mass_ratio,
            )
            paired_input_per_kg_h2 = (
                self.p.hydrogen_electrolyser_kwh_per_kg
                + self.p.hydrogen_compression_kwh_per_kg
                + self.p.co2_to_hydrogen_mass_ratio * self.p.co2_production_kwh_per_kg
                + self.p.co2_to_hydrogen_mass_ratio * self.p.co2_compression_kwh_per_kg
            )
            paired_h2_kg = min(paired_h2_room_kg, limit / paired_input_per_kg_h2)
            paired_bus = paired_h2_kg * paired_input_per_kg_h2
            self.soc += paired_h2_kg * self.p.hydrogen_lhv_kwh_per_kg
            self.co2_soc_kg += paired_h2_kg * self.p.co2_to_hydrogen_mass_ratio

            # CO2 is charged first in a matched reaction packet.  Once its vessel
            # is full, remaining surplus may make additional H2 for the fuel cell.
            remaining_bus = limit - paired_bus
            remaining_h2_room_kg = max(
                0.0,
                (self.p.capacity_kwh * self.capacity_fraction - self.soc)
                / self.p.hydrogen_lhv_kwh_per_kg,
            )
            remaining_h2_limit_kg = max(0.0, max_h2_charge_kg - paired_h2_kg)
            extra_h2_kg = min(
                remaining_h2_room_kg,
                remaining_h2_limit_kg,
                remaining_bus / (self.p.hydrogen_electrolyser_kwh_per_kg
                                 + self.p.hydrogen_compression_kwh_per_kg),
            )
            extra_bus = extra_h2_kg * (self.p.hydrogen_electrolyser_kwh_per_kg
                                       + self.p.hydrogen_compression_kwh_per_kg)
            self.soc += extra_h2_kg * self.p.hydrogen_lhv_kwh_per_kg
            return paired_bus + extra_bus
        if self.p.method == "hydrogen":
            bus_energy = min(
                bus_energy,
                max_h2_charge_kg * (self.p.hydrogen_electrolyser_kwh_per_kg
                                    + self.p.hydrogen_compression_kwh_per_kg),
            )
        charge_ratio = _carrier_kwh_per_charge_kwh(self.p)
        limit = bus_energy
        room = max(0.0, self.p.capacity_kwh * self.capacity_fraction - self.soc)
        accepted = min(limit, room / charge_ratio)
        self.soc += accepted * charge_ratio
        return accepted

    def discharge(self, demand: float, reserve_carrier_kwh: float = 0.0, *,
                  max_h2_output_kg: float = math.inf) -> float:
        if demand <= 0 or self.soc <= 0:
            return 0.0
        discharge_ratio = _discharge_kwh_per_carrier_kwh(self.p)
        usable_carrier = max(0.0, self.soc - reserve_carrier_kwh)
        h2_output_limit = (
            max_h2_output_kg * _net_h2_bus_kwh_per_kg(self.p)
            if self.p.method in {"hydrogen", "h2_co2"} else math.inf
        )
        delivered = min(
            demand, usable_carrier * discharge_ratio, h2_output_limit,
        )
        self.soc -= delivered / discharge_ratio
        return delivered

    def deliverable(self, reserve_carrier_kwh: float = 0.0, *,
                    max_h2_output_kg: float = math.inf) -> float:
        """Return bus energy that could be discharged this hour without mutating SOC."""
        if self.soc <= 0:
            return 0.0
        usable_carrier = max(0.0, self.soc - reserve_carrier_kwh)
        h2_output_limit = (
            max_h2_output_kg * _net_h2_bus_kwh_per_kg(self.p)
            if self.p.method in {"hydrogen", "h2_co2"} else math.inf
        )
        return min(
            usable_carrier * _discharge_kwh_per_carrier_kwh(self.p),
            h2_output_limit,
        )

    def reservable_hydrogen_kg(self, h2_required: float,
                               reserve_carrier_kwh: float = 0.0, *,
                               max_h2_output_kg: float = math.inf) -> float:
        if self.p.method not in {"hydrogen", "h2_co2"} or h2_required <= 0:
            return 0.0
        available_h2 = max(0.0, self.soc - reserve_carrier_kwh) / self.p.hydrogen_lhv_kwh_per_kg
        return min(h2_required, available_h2, max_h2_output_kg)

    def reservable_co2_kg(self, co2_required_kg: float) -> float:
        if self.p.method != "h2_co2" or co2_required_kg <= 0:
            return 0.0
        return min(co2_required_kg, self.co2_soc_kg)

    def feed_offsets(self, h2_required: float, co2_required_kg: float = 0.0,
                     reserve_carrier_kwh: float = 0.0, *,
                     max_h2_output_kg: float = math.inf) -> tuple[float, float, float]:
        """Return independently usable stored feed and avoided upstream energy."""
        stored_h2 = self.reservable_hydrogen_kg(
            h2_required, reserve_carrier_kwh,
            max_h2_output_kg=max_h2_output_kg,
        )
        stored_co2 = self.reservable_co2_kg(co2_required_kg)
        # Stored feed avoids fresh production and gains expander work, less the
        # electrical reheat that expansion requires.
        avoided = stored_h2 * (
            self.p.hydrogen_electrolyser_kwh_per_kg
            + self.p.hydrogen_expansion_kwh_per_kg
            - self.p.hydrogen_expansion_reheat_kwh_per_kg
        )
        if self.p.method == "h2_co2":
            avoided += stored_co2 * (
                self.p.co2_production_kwh_per_kg
                + self.p.co2_expansion_kwh_per_kg
                - self.p.co2_expansion_reheat_kwh_per_kg
            )
        return stored_h2, stored_co2, avoided

    def use_feed_inventory(self, h2_required: float, co2_required_kg: float = 0.0,
                           reserve_carrier_kwh: float = 0.0, *,
                           max_h2_output_kg: float = math.inf) -> tuple[float, float, float]:
        stored_h2, stored_co2, avoided = self.feed_offsets(
            h2_required, co2_required_kg, reserve_carrier_kwh,
            max_h2_output_kg=max_h2_output_kg,
        )
        self.soc -= stored_h2 * self.p.hydrogen_lhv_kwh_per_kg
        self.co2_soc_kg = max(0.0, self.co2_soc_kg - stored_co2)
        return stored_h2, stored_co2, avoided

    def use_hydrogen_direct(self, h2_required: float,
                            reserve_carrier_kwh: float = 0.0) -> tuple[float, float]:
        if self.p.method not in {"hydrogen", "h2_co2"} or h2_required <= 0:
            return 0.0, 0.0
        co2_required = (
            h2_required * self.p.co2_to_hydrogen_mass_ratio
            if self.p.method == "h2_co2" else 0.0
        )
        used, _, avoided = self.use_feed_inventory(
            h2_required, co2_required, reserve_carrier_kwh,
        )
        return used, avoided

    def direct_feed_avoided_kwh(self, h2_kg: float) -> float:
        return h2_kg * self.p.hydrogen_electrolyser_kwh_per_kg


def _co2_feed_requirement_kg(h2_required_kg: float, *stores: _StoreState) -> float:
    for store in stores:
        if store.p.method == "h2_co2":
            return h2_required_kg * store.p.co2_to_hydrogen_mass_ratio
    return 0.0


def _forecast_storage_trajectory_feasible(
        index: pd.DatetimeIndex, net_bus_kwh: np.ndarray,
        target_methane_kg: np.ndarray, h2_per_methane: float,
        short: _StoreState, long: _StoreState) -> bool:
    """Test a forecast trajectory using copies of the current storage states.

    Positive forecast balance charges short storage before long storage. Stored
    hydrogen is used directly for Sabatier feed before either store supplies an
    electrical deficit, matching the real dispatch priority. The first hour starts
    from the already-updated live SOC; later hours apply self-discharge. Faults
    are deliberately excluded from forecast commitment planning.
    """
    projected_short = _StoreState(short.p, short.soc, short.co2_soc_kg)
    projected_long = _StoreState(long.p, long.soc, long.co2_soc_kg)
    for offset, stamp in enumerate(index):
        if offset:
            projected_short.begin_hour(1.0)
            projected_long.begin_hour(1.0)

        h2_required = max(0.0, float(target_methane_kg[offset]) * h2_per_methane)
        co2_required = _co2_feed_requirement_kg(
            h2_required, projected_short, projected_long,
        )
        short_h2, short_co2, short_avoided = projected_short.use_feed_inventory(
            h2_required, co2_required,
        )
        h2_required -= short_h2
        co2_required -= short_co2
        _, _, long_avoided = projected_long.use_feed_inventory(
            h2_required, co2_required,
        )
        balance = float(net_bus_kwh[offset]) + short_avoided + long_avoided

        if balance >= 0:
            balance -= projected_short.charge(balance)
            projected_long.charge(balance)
            continue

        deficit = -balance
        deficit -= projected_short.discharge(deficit)
        deficit -= projected_long.discharge(deficit)
        if deficit > 1e-7:
            return False
    return True


def _planned_storage_soc_trajectory(
        forecast: pd.DataFrame, schedule: pd.DataFrame,
        short_storage: StorageParameters, long_storage: StorageParameters,
        short_initial_soc_kwh: float | None, long_initial_soc_kwh: float | None,
        plant: PlantParameters, h2_per_methane: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the forecast-only SOC reference used to assess forecast error."""
    short = _StoreState(short_storage, short_initial_soc_kwh)
    long = _StoreState(long_storage, long_initial_soc_kwh)
    generation = forecast["capacity_factor"].to_numpy(dtype=float) * plant.solar_farm_mw * 1000.0
    net = generation - schedule["target_total_load_kw"].to_numpy(dtype=float)
    target_methane = schedule["target_methane_kg_h"].to_numpy(dtype=float)
    short_soc = np.empty(len(forecast), dtype=float)
    long_soc = np.empty(len(forecast), dtype=float)
    for pos, stamp in enumerate(forecast.index):
        short.begin_hour(1.0)
        long.begin_hour(1.0)
        h2_required = max(0.0, target_methane[pos] * h2_per_methane)
        co2_required = _co2_feed_requirement_kg(h2_required, short, long)
        short_h2, short_co2, short_avoided = short.use_feed_inventory(
            h2_required, co2_required,
        )
        h2_required -= short_h2
        co2_required -= short_co2
        _, _, long_avoided = long.use_feed_inventory(h2_required, co2_required)
        balance = net[pos] + short_avoided + long_avoided
        if balance >= 0:
            balance -= short.charge(balance)
            long.charge(balance)
        else:
            balance = -balance
            balance -= short.discharge(balance)
            long.discharge(balance)
        short_soc[pos] = short.soc
        long_soc[pos] = long.soc
    return short_soc, long_soc


def _serve_electric_load(demand: float, pv: float, short: _StoreState, long: _StoreState, *,
                         short_reserve_kwh: float = 0.0,
                         long_reserve_kwh: float = 0.0,
                         long_h2_output_limit_kg: float = math.inf) -> tuple[float, float, float, float]:
    from_pv = min(demand, pv)
    remaining, pv_left = demand - from_pv, pv - from_pv
    from_short = short.discharge(remaining, short_reserve_kwh)
    remaining -= from_short
    from_long = long.discharge(
        remaining, long_reserve_kwh,
        max_h2_output_kg=long_h2_output_limit_kg,
    )
    return demand - (remaining - from_long), pv_left, from_short, from_long


def simulate_dispatch(actual: pd.DataFrame, forecast: pd.DataFrame, energy: PlantEnergyResult,
                      nominal: float, short_storage: StorageParameters, long_storage: StorageParameters,
                      plant: PlantParameters = PlantParameters(), thermal: ThermalParameters = ThermalParameters(),
                      strategy: StrategyConfig = StrategyConfig(), faults: Sequence[FaultEvent] = (), *,
                      short_initial_soc_kwh: float | None = None,
                      long_initial_soc_kwh: float | None = None,
                      progress: ProgressCallback | None = None,
                      progress_label: str = "Dispatch progress",
                      commitment_schedule: pd.DataFrame | None = None,
                      fault_capacities: Mapping[str, float] | None = None,
                      methane_delivery_kg_h: float | None = None) -> SimulationResult:
    """Dispatch storage-aware integer Sabatier commitments against actual weather."""
    actual = validate_hourly_profile(actual)
    forecast = validate_hourly_profile(forecast).reindex(actual.index)
    if forecast["capacity_factor"].isna().any():
        raise WeatherDataError("Forecast and actual timestamps do not align")
    fixed_commitment = commitment_schedule is not None
    if fixed_commitment:
        schedule = commitment_schedule.reindex(actual.index)
        if schedule["planned_reactor_count"].isna().any():
            raise WeatherDataError("Commitment schedule and actual timestamps do not align")
    else:
        schedule = build_target_schedule(forecast, nominal, energy, plant, thermal, strategy)
    short = _StoreState(short_storage, short_initial_soc_kwh)
    long = _StoreState(long_storage, long_initial_soc_kwh)
    planned_short_soc, planned_long_soc = _planned_storage_soc_trajectory(
        forecast, schedule, short_storage, long_storage,
        short_initial_soc_kwh, long_initial_soc_kwh, plant,
        energy.stoichiometry_kg_per_kg_ch4["H2"],
    )
    _, _, carb_c, calc_c = _thermal_inventory_and_capacity(nominal, plant)
    train_count = strategy.parallel_reactor_count
    unit_rate = nominal / train_count
    sab_c, sab_ua = sabatier_train_thermal_properties(nominal, thermal, train_count)
    sab_temperatures = np.full(train_count, plant.sabatier_temperature_k, dtype=float)
    t_carb = plant.carbonator_temperature_k
    t_calc = plant.calciner_temperature_k
    previous_count = 0
    h2_per_methane = energy.stoichiometry_kg_per_kg_ch4["H2"]
    co2_per_methane = energy.stoichiometry_kg_per_kg_ch4["CO2"]
    methane_expansion_kwh_per_kg = energy.gas_storage_work_kwh_per_kg.get(
        "methane_expansion", 0.0,
    )
    methane_expansion_reheat_kwh_per_kg = energy.gas_storage_work_kwh_per_kg.get(
        "methane_expansion_reheat", 0.0,
    )
    # Net bus contribution of the product expander: recovered work less the
    # electrical reheat it needs. May be negative, in which case it is a load.
    methane_expansion_net_kwh_per_kg = (
        methane_expansion_kwh_per_kg - methane_expansion_reheat_kwh_per_kg
    )
    buffered_methane = methane_delivery_kg_h is not None
    methane_delivery = max(0.0, float(methane_delivery_kg_h or 0.0))
    methane_expander_generation_kwh = methane_delivery * methane_expansion_kwh_per_kg
    methane_expander_reheat_kwh = methane_delivery * methane_expansion_reheat_kwh_per_kg
    methane_expander_net_kwh = methane_delivery * methane_expansion_net_kwh_per_kg
    fault_capacities = dict(fault_capacities or {})
    max_count_change = int(math.floor(plant.ramp_fraction_per_h * train_count + 1e-12))
    forecast_generation = forecast["capacity_factor"].to_numpy(dtype=float) * plant.solar_farm_mw * 1000.0
    reference_net = forecast_generation - schedule["target_total_load_kw"].to_numpy(dtype=float)
    reference_target = schedule["target_methane_kg_h"].to_numpy(dtype=float)
    current_day: pd.Timestamp | None = None
    committed_daily_count = 0
    storage_aware_round_up_days = 0

    numeric_columns = (
        "actual_cf", "forecast_cf", "generation_kwh",
        "continuous_reactor_equivalents", "planned_reactor_count", "actual_reactor_count",
        "target_methane_kg", "methane_kg", "methane_shortfall_kg",
        "methane_storage_inflow_kg", "methane_storage_outflow_kg",
        "methane_storage_compressor_load_kwh", "methane_storage_expander_generation_kwh",
        "methane_storage_expander_reheat_kwh",
        "thermal_load_kwh", "common_thermal_load_kwh", "sabatier_thermal_load_kwh",
        "startup_load_kwh", "common_startup_load_kwh", "sabatier_startup_load_kwh",
        "process_load_requested_kwh", "electricity_served_kwh",
        "short_charge_kwh", "short_discharge_kwh", "short_soc_kwh", "planned_short_soc_kwh",
        "long_charge_kwh", "long_discharge_kwh", "long_soc_kwh", "planned_long_soc_kwh",
        "long_h2_charge_kg", "long_co2_charge_kg", "long_h2_soc_kg", "long_co2_soc_kg",
        "long_fuel_cell_h2_kg", "direct_h2_kg", "direct_co2_kg",
        "long_direct_h2_kg", "long_direct_co2_kg",
        "long_h2_storage_compressor_load_kwh",
        "long_h2_storage_expander_generation_kwh",
        "long_h2_storage_expander_reheat_kwh",
        "long_co2_storage_compressor_load_kwh",
        "long_co2_storage_expander_generation_kwh",
        "long_co2_storage_expander_reheat_kwh",
        "curtailed_kwh", "carbonator_temperature_k",
        "calciner_temperature_k", "sabatier_mean_temperature_k",
        "sabatier_min_temperature_k",
        "battery_capacity_fraction", "hydrogen_capacity_fraction",
        "sabatier_capacity_fraction", "dac_capacity_fraction",
        "battery_fault_active", "hydrogen_fault_active",
        "sabatier_fault_active", "dac_fault_active",
        "short_inaccessible_kwh", "long_inaccessible_kwh",
        "fresh_h2_kg", "fresh_co2_kg",
        "energy_balance_residual_kwh",
    )
    row_data = {column: np.empty(len(actual), dtype=np.float64) for column in numeric_columns}
    state_data = np.empty(len(actual), dtype=object)
    warnings = list(energy.warnings)
    reported_period: tuple[int, int] | None = None

    for pos, stamp in enumerate(actual.index):
        period = (stamp.year, (stamp.month - 1) // 3 + 1)
        if period != reported_period:
            reported_period = period
            _report_progress(
                progress,
                f"{progress_label}: {pos + 1:,}/{len(actual):,} hours ({stamp.year} Q{period[1]}).",
            )
        actual_cf = float(actual["capacity_factor"].iloc[pos])
        ambient_value = actual["ambient_temperature_k"].iloc[pos]
        ambient = thermal.ambient_fallback_k if pd.isna(ambient_value) else float(ambient_value)
        generated = actual_cf * plant.solar_farm_mw * 1000.0
        battery_fraction = _fault_fraction_at(stamp, "battery", faults)
        hydrogen_fraction = _fault_fraction_at(stamp, "hydrogen", faults)
        sabatier_fraction = _fault_fraction_at(stamp, "sabatier", faults)
        dac_fraction = _fault_fraction_at(stamp, "dac", faults)
        battery_active = _fault_active_at(stamp, "battery", faults)
        hydrogen_active = _fault_active_at(stamp, "hydrogen", faults)
        sabatier_active = _fault_active_at(stamp, "sabatier", faults)
        dac_active = _fault_active_at(stamp, "dac", faults)
        short_capacity_fraction = battery_fraction if short.p.method == "battery" else 1.0
        long_capacity_fraction = battery_fraction if long.p.method == "battery" else 1.0
        short.begin_hour(short_capacity_fraction)
        long.begin_hour(long_capacity_fraction)

        def active_fault_limit(name: str, fraction: float, fallback: float) -> float:
            if fraction >= 1.0:
                return math.inf
            reference = float(fault_capacities.get(name, fallback))
            return max(0.0, reference * fraction)

        day = stamp.normalize()
        if current_day is None or day != current_day:
            current_day = day
            # Bound unconditionally: the round-up guard below only avoids an
            # UnboundLocalError today by short-circuiting on the same condition.
            lower_count = upper_count = 0
            if fixed_commitment or strategy.reactor_scheduling_mode == "seasonal":
                committed_daily_count = int(schedule["planned_reactor_count"].iloc[pos])
            else:
                equivalents = float(schedule["continuous_reactor_equivalents"].iloc[pos])
                lower_count = min(train_count, max(0, math.floor(equivalents)))
                upper_count = min(train_count, max(0, math.ceil(equivalents)))
                committed_daily_count = lower_count
            if (not fixed_commitment and strategy.reactor_scheduling_mode != "seasonal"
                    and upper_count > lower_count):
                day_end = int(actual.index.searchsorted(day + pd.Timedelta(days=1)))
                horizon_end = int(actual.index.searchsorted(
                    stamp + pd.Timedelta(days=strategy.storage_planning_lookahead_days)
                ))
                horizon_end = max(day_end, min(len(actual), horizon_end))
                projected_net = reference_net[pos:horizon_end].copy()
                projected_target = reference_target[pos:horizon_end].copy()
                current_hours = day_end - pos
                if strategy.short_strategy == "through_night":
                    candidate_active = np.ones(current_hours, dtype=bool)
                else:
                    candidate_active = schedule["daylight"].iloc[pos:day_end].to_numpy(dtype=bool)
                candidate_target = candidate_active.astype(float) * upper_count * unit_rate
                target_delta = candidate_target - projected_target[:current_hours]
                projected_target[:current_hours] = candidate_target
                projected_net[:current_hours] -= target_delta * energy.e_req_kwh_per_kg_ch4
                if _forecast_storage_trajectory_feasible(
                        actual.index[pos:horizon_end], projected_net, projected_target,
                        h2_per_methane, short, long):
                    committed_daily_count = upper_count
                    storage_aware_round_up_days += 1

        planned_count = committed_daily_count
        if fixed_commitment or strategy.reactor_scheduling_mode == "seasonal":
            planned_count = int(schedule["planned_reactor_count"].iloc[pos])
        elif strategy.short_strategy != "through_night" and not bool(schedule["daylight"].iloc[pos]):
            planned_count = 0
        if max_count_change < train_count:
            planned_count = min(planned_count, previous_count + max_count_change)
            planned_count = max(0, max(planned_count, previous_count - max_count_change))

        # A fixed commitment comes with a separately planned SOC trajectory that
        # must be protected for that case.  When dispatch is building its own
        # daily commitment, the storage-aware planner has already made the
        # feasibility decision from the live inventories; reserving the initial
        # forecast trajectory again would incorrectly reject a supportable ceiling.
        planned_short_reserve = max(0.0, float(planned_short_soc[pos])) if fixed_commitment else 0.0
        planned_long_reserve = max(0.0, float(planned_long_soc[pos])) if fixed_commitment else 0.0
        order = np.argsort(-sab_temperatures)

        def demands(candidate_count: int) -> dict[str, float | np.ndarray]:
            active_ids = order[:candidate_count]
            inactive_ids = order[candidate_count:]
            common_operating = candidate_count > 0 or strategy.short_strategy != "hard_shutdown"
            if common_operating:
                common_thermal = (
                    thermal.carbonator_ua_kw_per_k * max(0.0, t_carb - ambient)
                    + thermal.calciner_ua_kw_per_k * max(0.0, t_calc - ambient)
                    + thermal.fixed_hot_auxiliary_kw
                )
                common_startup = (
                    max(0.0, carb_c * (plant.carbonator_temperature_k - t_carb))
                    + max(0.0, calc_c * (plant.calciner_temperature_k - t_calc))
                )
            else:
                common_thermal = common_startup = 0.0
            if strategy.short_strategy == "hard_shutdown":
                sab_thermal = 0.0
                sab_startup = float(np.maximum(
                    0.0, sab_c * (plant.sabatier_temperature_k - sab_temperatures[active_ids])
                ).sum()) if candidate_count else 0.0
            else:
                sab_thermal = float(
                    sab_ua * max(0.0, plant.sabatier_temperature_k - ambient) * len(inactive_ids)
                )
                sab_startup = 0.0
            desired_methane = candidate_count * unit_rate * sabatier_fraction
            desired_h2 = desired_methane * h2_per_methane
            desired_co2 = desired_methane * co2_per_methane
            h2_handling_limit = active_fault_limit(
                "stored_h2_handling_kg_h", hydrogen_fraction, desired_h2,
            )
            stored_h2_available = min(
                h2_handling_limit,
                short.reservable_hydrogen_kg(desired_h2, planned_short_reserve)
                + long.reservable_hydrogen_kg(desired_h2, planned_long_reserve),
            )
            stored_co2_available = (
                short.reservable_co2_kg(desired_co2)
                + long.reservable_co2_kg(desired_co2)
            )
            fresh_h2_limit = active_fault_limit(
                "process_h2_kg_h", hydrogen_fraction, desired_h2,
            )
            fresh_co2_limit = active_fault_limit(
                "dac_co2_kg_h", dac_fraction, desired_co2,
            )
            methane = min(
                desired_methane,
                ((stored_h2_available + fresh_h2_limit) / h2_per_methane
                 if h2_per_methane else desired_methane),
                ((stored_co2_available + fresh_co2_limit) / co2_per_methane
                 if co2_per_methane else desired_methane),
            )
            target_h2 = methane * h2_per_methane
            target_co2 = methane * co2_per_methane
            direct_short, direct_short_co2, short_avoided = short.feed_offsets(
                target_h2, target_co2, planned_short_reserve,
                max_h2_output_kg=h2_handling_limit,
            )
            remaining_handler = max(0.0, h2_handling_limit - direct_short)
            direct_long, direct_long_co2, long_avoided = long.feed_offsets(
                target_h2 - direct_short,
                target_co2 - direct_short_co2,
                planned_long_reserve,
                max_h2_output_kg=remaining_handler,
            )
            direct_h2 = direct_short + direct_long
            fresh_h2 = max(0.0, target_h2 - direct_h2)
            fresh_co2 = max(0.0, target_co2 - direct_short_co2 - direct_long_co2)
            # E_req contains net CH4 pressure work, i.e. compression plus expander
            # reheat less recovered expansion. Restore the net expansion term here so
            # process load carries only the production-proportional vessel-inlet
            # compression duty; the constant vessel outflow receives the expander's
            # net credit, reheat included, separately.
            process = max(0.0, methane * (
                energy.e_req_kwh_per_kg_ch4
                + (methane_expansion_net_kwh_per_kg if buffered_methane else 0.0))
                          - short_avoided - long_avoided)
            short_reserve = max(
                planned_short_reserve,
                direct_short * short.p.hydrogen_lhv_kwh_per_kg,
            )
            long_reserve = max(
                planned_long_reserve,
                direct_long * long.p.hydrogen_lhv_kwh_per_kg,
            )
            total = max(0.0, common_thermal + common_startup + sab_thermal
                        + sab_startup + process - methane_expander_net_kwh)
            remaining_h2_handler = max(0.0, h2_handling_limit - direct_h2)
            fuel_cell_limit = min(
                remaining_h2_handler,
                active_fault_limit(
                    "fuel_cell_h2_kg_h", hydrogen_fraction, remaining_h2_handler,
                ),
            )
            available = (
                generated
                + short.deliverable(short_reserve)
                + long.deliverable(
                    long_reserve,
                    max_h2_output_kg=fuel_cell_limit,
                )
            )
            return {
                "active_ids": active_ids, "inactive_ids": inactive_ids,
                "common_thermal": common_thermal, "common_startup": common_startup,
                "sab_thermal": sab_thermal, "sab_startup": sab_startup,
                "methane": methane, "direct_short": direct_short,
                "direct_long": direct_long,
                "direct_short_co2": direct_short_co2,
                "direct_long_co2": direct_long_co2,
                "fresh_h2": fresh_h2, "fresh_co2": fresh_co2,
                "fuel_cell_h2_limit": fuel_cell_limit,
                "short_reserve": short_reserve,
                "long_reserve": long_reserve, "process": process, "total": total,
                "feasible": available + 1e-9 >= total,
            }

        selected: dict[str, float | np.ndarray] | None = None
        actual_count = 0
        for candidate in range(planned_count, -1, -1):
            candidate_demands = demands(candidate)
            if bool(candidate_demands["feasible"]):
                selected = candidate_demands
                actual_count = candidate
                break
        if selected is None:
            # Limping heat can itself be unavailable. Keep the zero-train demand so
            # the partial thermal service and forced shutdown remain visible.
            selected = demands(0)
        total_requested = float(selected["total"])
        served, pv, short_discharge, long_discharge = _serve_electric_load(
            total_requested, generated, short, long,
            short_reserve_kwh=float(selected["short_reserve"]),
            long_reserve_kwh=float(selected["long_reserve"]),
            long_h2_output_limit_kg=float(selected["fuel_cell_h2_limit"]),
        )
        fully_served = served + 1e-8 >= total_requested
        if not fully_served:
            actual_count = 0

        direct_short_used = direct_long_used = 0.0
        direct_short_co2_used = direct_long_co2_used = 0.0
        methane = float(selected["methane"]) if fully_served else 0.0
        if methane > 0:
            direct_short_used, direct_short_co2_used, _ = short.use_feed_inventory(
                float(selected["direct_short"]),
                float(selected["direct_short_co2"]),
                planned_short_reserve,
            )
            direct_long_used, direct_long_co2_used, _ = long.use_feed_inventory(
                float(selected["direct_long"]),
                float(selected["direct_long_co2"]),
                planned_long_reserve,
            )

        active_ids = np.asarray(selected["active_ids"], dtype=int) if fully_served else np.array([], dtype=int)
        inactive_ids = np.setdiff1d(np.arange(train_count), active_ids, assume_unique=False)
        if strategy.short_strategy == "hard_shutdown":
            if len(active_ids):
                sab_temperatures[active_ids] = plant.sabatier_temperature_k
            if len(inactive_ids):
                decay = math.exp(-sab_ua / sab_c)
                sab_temperatures[inactive_ids] = ambient + (sab_temperatures[inactive_ids] - ambient) * decay
            if actual_count > 0:
                t_carb, t_calc = plant.carbonator_temperature_k, plant.calciner_temperature_k
                actual_state = "operating"
            else:
                t_carb = ambient + (t_carb - ambient) * math.exp(-thermal.carbonator_ua_kw_per_k / carb_c)
                t_calc = ambient + (t_calc - ambient) * math.exp(-thermal.calciner_ua_kw_per_k / calc_c)
                actual_state = "cooling" if fully_served else "forced_shutdown"
        elif fully_served:
            sab_temperatures[:] = plant.sabatier_temperature_k
            t_carb, t_calc = plant.carbonator_temperature_k, plant.calciner_temperature_k
            actual_state = "operating" if actual_count > 0 else "limping"
        else:
            sab_temperatures[:] = ambient + (sab_temperatures - ambient) * math.exp(-sab_ua / sab_c)
            t_carb = ambient + (t_carb - ambient) * math.exp(-thermal.carbonator_ua_kw_per_k / carb_c)
            t_calc = ambient + (t_calc - ambient) * math.exp(-thermal.calciner_ua_kw_per_k / calc_c)
            actual_state = "forced_shutdown"

        short_charge = short.charge(pv)
        pv -= short_charge
        long_h2_before = long.soc
        long_co2_before = long.co2_soc_kg
        prospective_h2_charge = (
            pv / (long.p.hydrogen_electrolyser_kwh_per_kg
                  + long.p.hydrogen_compression_kwh_per_kg)
            if long.p.method in {"hydrogen", "h2_co2"} else math.inf
        )
        storage_h2_charge_limit = min(
            active_fault_limit(
                "storage_h2_charge_kg_h", hydrogen_fraction,
                prospective_h2_charge,
            ),
            active_fault_limit(
                "stored_h2_handling_kg_h", hydrogen_fraction,
                prospective_h2_charge,
            ),
        )
        process_fresh_co2 = float(selected["fresh_co2"]) if fully_served else 0.0
        total_dac_limit = active_fault_limit(
            "dac_co2_kg_h", dac_fraction, process_fresh_co2,
        )
        storage_co2_charge_limit = max(0.0, total_dac_limit - process_fresh_co2)
        long_charge = long.charge(
            pv,
            max_h2_charge_kg=storage_h2_charge_limit,
            max_co2_charge_kg=storage_co2_charge_limit,
        )
        pv -= long_charge
        if long.p.method in {"hydrogen", "h2_co2"}:
            long_h2_charge_kg = max(
                0.0,
                (long.soc - long_h2_before) / long.p.hydrogen_lhv_kwh_per_kg,
            )
            long_h2_soc_kg = long.soc / long.p.hydrogen_lhv_kwh_per_kg
            long_fuel_cell_h2_kg = long_discharge / _net_h2_bus_kwh_per_kg(long.p)
        else:
            long_h2_charge_kg = long_h2_soc_kg = long_fuel_cell_h2_kg = 0.0
        long_co2_charge_kg = max(0.0, long.co2_soc_kg - long_co2_before)
        long_co2_soc_kg = long.co2_soc_kg
        direct_co2_kg = direct_short_co2_used + direct_long_co2_used
        fresh_h2_kg = float(selected["fresh_h2"]) if fully_served else 0.0
        fresh_co2_kg = float(selected["fresh_co2"]) if fully_served else 0.0
        curtailed = max(0.0, pv)
        target_methane = planned_count * unit_rate
        thermal_load = float(selected["common_thermal"]) + float(selected["sab_thermal"])
        startup_load = float(selected["common_startup"]) + float(selected["sab_startup"])
        residual = (
            generated + short_discharge + long_discharge
            - short_charge - long_charge - curtailed - served
        )
        row_values = (
            actual_cf, float(forecast["capacity_factor"].iloc[pos]), generated,
            float(schedule["continuous_reactor_equivalents"].iloc[pos]),
            planned_count, actual_count,
            target_methane, methane, max(0.0, target_methane - methane),
            methane, methane_delivery if buffered_methane else methane,
            methane * energy.gas_storage_work_kwh_per_kg.get("methane_compression", 0.0),
            (methane_expander_generation_kwh if buffered_methane
             else methane * methane_expansion_kwh_per_kg),
            (methane_expander_reheat_kwh if buffered_methane
             else methane * methane_expansion_reheat_kwh_per_kg),
            thermal_load, float(selected["common_thermal"]), float(selected["sab_thermal"]),
            startup_load, float(selected["common_startup"]), float(selected["sab_startup"]),
            float(selected["process"]), served,
            short_charge, short_discharge, short.soc, planned_short_soc[pos],
            long_charge, long_discharge, long.soc, planned_long_soc[pos],
            long_h2_charge_kg, long_co2_charge_kg, long_h2_soc_kg, long_co2_soc_kg,
            long_fuel_cell_h2_kg, direct_short_used + direct_long_used, direct_co2_kg,
            direct_long_used, direct_long_co2_used,
            long_h2_charge_kg * long.p.hydrogen_compression_kwh_per_kg,
            (direct_long_used + long_fuel_cell_h2_kg)
            * long.p.hydrogen_expansion_kwh_per_kg,
            (direct_long_used + long_fuel_cell_h2_kg)
            * long.p.hydrogen_expansion_reheat_kwh_per_kg,
            long_co2_charge_kg * long.p.co2_compression_kwh_per_kg,
            direct_long_co2_used * long.p.co2_expansion_kwh_per_kg,
            direct_long_co2_used * long.p.co2_expansion_reheat_kwh_per_kg,
            curtailed, t_carb, t_calc,
            float(sab_temperatures.mean()), float(sab_temperatures.min()),
            battery_fraction, hydrogen_fraction, sabatier_fraction, dac_fraction,
            float(battery_active), float(hydrogen_active),
            float(sabatier_active), float(dac_active),
            short.inaccessible_soc, long.inaccessible_soc,
            fresh_h2_kg, fresh_co2_kg,
            residual,
        )
        if len(row_values) != len(numeric_columns):
            raise AssertionError("Dispatch output columns and row values are misaligned")
        for column, value in zip(numeric_columns, row_values):
            row_data[column][pos] = value
        state_data[pos] = actual_state
        previous_count = actual_count

    _report_progress(progress, f"{progress_label}: hourly stepping complete; assembling results.")
    row_data["state"] = state_data
    hourly = pd.DataFrame(row_data, index=actual.index, copy=False)
    daily = hourly.resample("D").agg({
        "generation_kwh": "sum", "target_methane_kg": "sum", "methane_kg": "sum",
        "methane_shortfall_kg": "sum", "curtailed_kwh": "sum",
        "methane_storage_inflow_kg": "sum", "methane_storage_outflow_kg": "sum",
        "methane_storage_compressor_load_kwh": "sum",
        "methane_storage_expander_generation_kwh": "sum",
        "methane_storage_expander_reheat_kwh": "sum",
        "long_h2_storage_compressor_load_kwh": "sum",
        "long_h2_storage_expander_generation_kwh": "sum",
        "long_h2_storage_expander_reheat_kwh": "sum",
        "long_co2_storage_compressor_load_kwh": "sum",
        "long_co2_storage_expander_generation_kwh": "sum",
        "long_co2_storage_expander_reheat_kwh": "sum",
        "short_soc_kwh": "last", "long_soc_kwh": "last",
        "planned_reactor_count": "max", "actual_reactor_count": "max",
        "energy_balance_residual_kwh": "sum",
    })
    years = len(hourly) / 8760.0
    methane_total = float(hourly["methane_kg"].sum())
    target_total = float(hourly["target_methane_kg"].sum())
    generated_total = float(hourly["generation_kwh"].sum())
    train_shortfall = hourly["planned_reactor_count"] - hourly["actual_reactor_count"]
    forecast_ceiling_violations = int((train_shortfall < -1e-9).sum())
    if forecast_ceiling_violations:
        raise ModelError(
            "Dispatch activated more Sabatier trains than the forecast commitment allowed"
        )
    metrics: dict[str, float | int | str | None] = {
        "simulated_hours": len(hourly),
        "average_annual_methane_kg": methane_total / years if years else 0.0,
        "average_annual_methane_shortfall_kg": float(hourly["methane_shortfall_kg"].sum()) / years if years else 0.0,
        "methane_total_kg": methane_total, "target_methane_total_kg": target_total,
        "plant_utilisation": methane_total / target_total if target_total else 0.0,
        "curtailed_energy_kwh": float(hourly["curtailed_kwh"].sum()),
        "curtailment_fraction": float(hourly["curtailed_kwh"].sum()) / generated_total if generated_total else 0.0,
        "curtailment_days": int((daily["curtailed_kwh"] > 1e-9).sum()),
        "forced_shutdown_hours": int((hourly["state"] == "forced_shutdown").sum()),
        "reactor_train_shortfall_hours": float(train_shortfall.clip(lower=0).sum()),
        "storage_aware_round_up_days": storage_aware_round_up_days,
        "forecast_reactor_ceiling_violations": forecast_ceiling_violations,
        "max_abs_energy_balance_residual_kwh": float(hourly["energy_balance_residual_kwh"].abs().max()),
        "max_short_inaccessible_kwh": short.max_inaccessible_soc,
        "max_long_inaccessible_kwh": long.max_inaccessible_soc,
    }
    for component in ("battery", "hydrogen", "sabatier", "dac"):
        active_column = f"{component}_fault_active"
        capacity_column = f"{component}_capacity_fraction"
        metrics[f"{component}_faulted_hours"] = int((hourly[active_column] > 0.5).sum())
        metrics[f"{component}_derated_hours"] = int(
            ((hourly[active_column] > 0.5) & (hourly[capacity_column] < 1.0 - 1e-12)).sum()
        )
    if long.p.method == "h2_co2":
        matched_co2 = hourly["long_h2_soc_kg"] * long.p.co2_to_hydrogen_mass_ratio
        unmatched_co2 = (hourly["long_co2_soc_kg"] - matched_co2).clip(lower=0.0)
        extra_h2 = (
            hourly["long_h2_soc_kg"]
            - hourly["long_co2_soc_kg"] / long.p.co2_to_hydrogen_mass_ratio
        ).clip(lower=0.0)
        metrics.update({
            "unmatched_co2_inventory_hours": int((unmatched_co2 > 1e-9).sum()),
            "max_unmatched_co2_kg": float(unmatched_co2.max()),
            "extra_hydrogen_inventory_hours": int((extra_h2 > 1e-9).sum()),
            "max_extra_hydrogen_kg": float(extra_h2.max()),
        })
    if metrics["max_abs_energy_balance_residual_kwh"] > 1e-6:
        warnings.append("Hourly bus energy balance residual exceeds numerical tolerance.")
    _report_progress(progress, f"{progress_label}: summaries complete.")
    raw_events = [_fault_event_dict(event) for event in faults]
    merged_intervals = _merged_fault_intervals(faults, horizon=actual.index)
    schedule_digest = sha256(
        json.dumps(raw_events, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return SimulationResult(
        hourly, daily, metrics, warnings,
        {"strategy": asdict(strategy), "short_storage": asdict(short_storage),
         "long_storage": asdict(long_storage),
         "fault_events": raw_events,
         "fault_effective_intervals": merged_intervals,
         "fault_schedule_digest": schedule_digest,
         "fault_generator_version": _FAULT_GENERATOR_VERSION,
         "fault_capacities": fault_capacities},
    )


def _simulate_dispatch_with_methane_storage(*args, **kwargs) -> SimulationResult:
    """Iterate constant CH4 delivery and recovered expansion power to consistency.

    ``vessel`` short-circuits that iteration. A plant whose tank was specified before
    the run does not get to discover its own delivery rate from what it turns out to
    produce: the rate was fixed at design time, and the run either keeps up with it or
    does not.
    """
    vessel = kwargs.pop("vessel", None)
    strategy = kwargs.get("strategy", args[8] if len(args) >= 9 else StrategyConfig())
    if not strategy.product_storage:
        # Nothing to converge on: delivery is whatever was made that hour, and
        # simulate_dispatch already reads an absent delivery rate as unbuffered.
        result = simulate_dispatch(*args, **kwargs)
        _attach_methane_storage_profile(
            result,
            kwargs.get("energy", args[2] if len(args) >= 3 else None),
            kwargs.get("plant", args[6] if len(args) >= 7 else PlantParameters()),
            buffered=False,
        )
        result.metadata["methane_storage"]["delivery_dispatch_iterations"] = 1
        return result
    if vessel is not None:
        energy = kwargs.get("energy", args[2] if len(args) >= 3 else None)
        if energy is None:
            raise ValueError("Plant energy result is required for methane storage dispatch")
        plant = kwargs.get("plant", args[6] if len(args) >= 7 else PlantParameters())
        result = simulate_dispatch(
            *args, **kwargs,
            methane_delivery_kg_h=float(vessel["continuous_delivery_kg_h"]),
        )
        _attach_methane_storage_profile(result, energy, plant, vessel=vessel,
                                        capacity_factor=strategy.f_socp_long)
        result.metadata["methane_storage"]["delivery_dispatch_iterations"] = 1
        return result
    delivery = 0.0
    result = None
    iterations = 0
    for iterations in range(1, 5):
        result = simulate_dispatch(*args, **kwargs, methane_delivery_kg_h=delivery)
        updated = float(result.hourly["methane_kg"].mean())
        tolerance = max(1e-8, 1e-7 * max(1.0, updated))
        if abs(updated - delivery) <= tolerance:
            break
        delivery = updated
    assert result is not None
    # Ensure the recorded outflow and expander credit use the final fixed-point value.
    if abs(float(result.hourly["methane_storage_outflow_kg"].iloc[0]) - updated) > tolerance:
        result = simulate_dispatch(*args, **kwargs, methane_delivery_kg_h=updated)
        iterations += 1
    energy = kwargs.get("energy", args[2] if len(args) >= 3 else None)
    if energy is None:
        raise ValueError("Plant energy result is required for methane storage dispatch")
    plant = kwargs.get("plant", args[6] if len(args) >= 7 else PlantParameters())
    _attach_methane_storage_profile(result, energy, plant,
                                    capacity_factor=strategy.f_socp_long)
    result.metadata["methane_storage"]["delivery_dispatch_iterations"] = iterations
    return result


# Economics, reporting, and orchestration --------------------------------------------


def load_costs(path: str | Path = "costs.json") -> dict[str, Any]:
    costs = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "financial", "solar", "battery", "methane_storage",
        "hydrogen_storage", "co2_storage", "plant_units", "plant_opex",
    }
    missing = required - set(costs)
    if missing:
        raise ModelError(f"Cost file is missing sections: {sorted(missing)}")
    return costs


def _scaled_cost(reference_cost: float, capacity: float, reference_capacity: float, exponent: float) -> float:
    return 0.0 if capacity <= 0 else reference_cost * (capacity / reference_capacity) ** exponent


def _capital_recovery_factor(rate: float, years: int) -> float:
    if years <= 0:
        raise ValueError("Project life must be positive")
    return 1 / years if rate == 0 else rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def size_methane_storage(methane_output_kg: pd.Series) -> dict[str, float]:
    """Size a lossless cyclic buffer for constant methane delivery at mean output."""
    output = pd.Series(methane_output_kg, copy=False).astype(float)
    if output.isna().any() or not np.isfinite(output.to_numpy()).all():
        raise ValueError("Methane output must contain only finite values")
    if (output < -1e-12).any():
        raise ValueError("Methane output cannot be negative")
    if output.empty:
        return {
            "capacity_kg": 0.0,
            "initial_inventory_kg": 0.0,
            "continuous_delivery_kg_h": 0.0,
            "minimum_output_kg_h": 0.0,
            "maximum_output_kg_h": 0.0,
        }
    output = output.clip(lower=0.0)
    delivery = float(output.mean())
    cumulative_imbalance = np.concatenate((
        np.array([0.0]),
        np.cumsum(output.to_numpy(dtype=float) - delivery),
    ))
    minimum = float(cumulative_imbalance.min())
    maximum = float(cumulative_imbalance.max())
    return {
        "capacity_kg": max(0.0, maximum - minimum),
        "initial_inventory_kg": max(0.0, -minimum),
        "continuous_delivery_kg_h": delivery,
        "minimum_output_kg_h": float(output.min()),
        "maximum_output_kg_h": float(output.max()),
    }


def _run_fixed_methane_vessel(output: np.ndarray, sizing: dict[str, float]
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run an already-built vessel against output it was not sized from.

    Sizing a vessel from the same series it then buffers makes the limits vacuous:
    the trajectory cannot leave [0, capacity] by construction. A vessel specified
    before the evaluation period has no such guarantee, so it is stepped hourly and
    both limits bite — it overflows when production runs ahead of the delivery rate
    for longer than it was built for, and it runs dry when production falls behind.

    Returns inventory, delivered and spilled, all per hour.
    """
    capacity = float(sizing["capacity_kg"])
    rate = float(sizing["continuous_delivery_kg_h"])
    soc = min(float(sizing["initial_inventory_kg"]), capacity)
    inventory = np.empty(len(output), dtype=float)
    delivered = np.empty(len(output), dtype=float)
    spilled = np.empty(len(output), dtype=float)
    for position, made in enumerate(output):
        soc += made
        over = soc - capacity
        if over > 0.0:
            soc = capacity
        else:
            over = 0.0
        out = rate if soc >= rate else soc
        soc -= out
        inventory[position], delivered[position], spilled[position] = soc, out, over
    return inventory, delivered, spilled


def _attach_methane_storage_profile(
        simulation: SimulationResult, energy: PlantEnergyResult,
        plant: PlantParameters, *, buffered: bool = True,
        vessel: dict[str, float] | None = None,
        capacity_factor: float = 1.0) -> dict[str, float]:
    """Attach the delivery profile and vessel inventory to a simulation.

    With ``buffered`` false the plant has no product vessel: delivery is whatever was
    made that hour, the inventory stays empty, and the compression and expansion terms
    that only exist to fill and empty a vessel all fall away. Every field is still
    populated, so sizing, economics and the flow diagram read it unchanged.

    ``vessel`` supplies a sizing settled before this run — the design study over the
    build-year record — instead of deriving one from this run's own output. The vessel
    is then a fixed asset that can overflow or run dry, and methane it cannot hold is
    curtailed.

    ``capacity_factor`` is f_SOCP applied to the product vessel exactly as it is to the
    long-term energy store: the design study says how large the tank needs to be, and
    this is the fraction of that actually built. Below one the tank cannot hold the
    swing it was designed for and spills; the delivery commitment is unchanged, because
    building a smaller tank does not reduce what was promised at the gate.
    """
    output = simulation.hourly["methane_kg"].astype(float).clip(lower=0.0)
    spill = None
    if buffered:
        design = dict(vessel) if vessel is not None else size_methane_storage(
            simulation.hourly["methane_kg"])
        sizing = dict(design)
        sizing["design_capacity_kg"] = design["capacity_kg"]
        sizing["capacity_kg"] = design["capacity_kg"] * capacity_factor
        sizing["initial_inventory_kg"] = min(
            design["initial_inventory_kg"] * capacity_factor, sizing["capacity_kg"])
    if buffered and (vessel is not None or capacity_factor != 1.0):
        inventory, delivered, spill = _run_fixed_methane_vessel(
            output.to_numpy(dtype=float), sizing,
        )
        index = simulation.hourly.index
        inventory = pd.Series(inventory, index=index)
        delivery = pd.Series(delivered, index=index)
        spill = pd.Series(spill, index=index)
    elif buffered:
        # A tank built exactly to its own study cannot leave [0, capacity], so the
        # cheap cumulative form is equivalent and stays the default path.
        delivery = sizing["continuous_delivery_kg_h"]
        inventory = (
            sizing["initial_inventory_kg"] + (output - delivery).cumsum()
        ).clip(lower=0.0, upper=sizing["capacity_kg"])
    else:
        sizing = {
            "capacity_kg": 0.0, "initial_inventory_kg": 0.0,
            "continuous_delivery_kg_h": float(output.mean()) if len(output) else 0.0,
            "minimum_output_kg_h": float(output.min()) if len(output) else 0.0,
            "maximum_output_kg_h": float(output.max()) if len(output) else 0.0,
        }
        delivery = output
        inventory = output * 0.0
    simulation.hourly["methane_delivery_kg"] = delivery
    simulation.hourly["methane_storage_inventory_kg"] = inventory
    simulation.hourly["methane_storage_inflow_kg"] = (
        (output - spill) if spill is not None else (output if buffered else output * 0.0)
    )
    simulation.hourly["methane_storage_outflow_kg"] = delivery
    # Methane the vessel had no room for. The plant made it and then threw it away, so
    # the energy that went into it was wasted exactly as surplus generation is, and it
    # joins curtailment rather than sitting in a column of its own that nothing totals.
    simulation.hourly["methane_curtailed_kg"] = (
        spill if spill is not None else output * 0.0
    )
    if (spill is not None and float(spill.sum()) > 0.0
            and "curtailed_kwh" in simulation.hourly):
        simulation.hourly["curtailed_kwh"] = (
            simulation.hourly["curtailed_kwh"].astype(float)
            + spill * energy.e_req_kwh_per_kg_ch4
        )
    compression = energy.gas_storage_work_kwh_per_kg.get("methane_compression", 0.0) if buffered else 0.0
    expansion = energy.gas_storage_work_kwh_per_kg.get("methane_expansion", 0.0) if buffered else 0.0
    reheat = energy.gas_storage_work_kwh_per_kg.get("methane_expansion_reheat", 0.0) if buffered else 0.0
    simulation.hourly["methane_storage_compressor_load_kwh"] = output * compression
    simulation.hourly["methane_storage_expander_generation_kwh"] = delivery * expansion
    simulation.hourly["methane_storage_expander_reheat_kwh"] = delivery * reheat
    if isinstance(simulation.daily.index, pd.DatetimeIndex):
        simulation.daily["methane_delivery_kg"] = (
            simulation.hourly["methane_delivery_kg"].resample("D").sum()
        )
        simulation.daily["methane_storage_inventory_kg"] = inventory.resample("D").last()
    delivered_total = float(simulation.hourly["methane_delivery_kg"].sum())
    curtailed_methane = float(simulation.hourly["methane_curtailed_kg"].sum())
    years = (simulation.metrics.get("simulated_hours") or len(simulation.hourly)) / 8760.0
    simulation.metrics.update({
        # What leaves the gate is what can be sold, so it is what LCOM is divided by.
        # With a vessel sized from this run's own output the two are equal; they part
        # company only when the vessel was specified in advance and could not hold
        # everything the plant went on to make.
        "average_annual_methane_kg": delivered_total / years if years else 0.0,
        "average_annual_methane_produced_kg": (
            float(simulation.hourly["methane_kg"].sum()) / years if years else 0.0),
        "average_annual_methane_curtailed_kg": curtailed_methane / years if years else 0.0,
        "methane_curtailed_total_kg": curtailed_methane,
        **({"curtailed_energy_kwh": float(simulation.hourly["curtailed_kwh"].sum())}
           if "curtailed_kwh" in simulation.hourly else {}),
        "methane_storage_capacity_kg": sizing["capacity_kg"],
        # What the design study asked for, before f_SOCP decided how much to build.
        "methane_storage_design_capacity_kg": sizing.get(
            "design_capacity_kg", sizing["capacity_kg"]),
        "continuous_methane_delivery_kg_h": (
            float(np.mean(delivery)) if hasattr(delivery, "__len__") else delivery),
        "methane_storage_compressor_energy_kwh": float(
            simulation.hourly["methane_storage_compressor_load_kwh"].sum()
        ),
        "methane_storage_expander_generation_kwh": float(
            simulation.hourly["methane_storage_expander_generation_kwh"].sum()
        ),
        "methane_storage_expander_reheat_kwh": float(
            simulation.hourly["methane_storage_expander_reheat_kwh"].sum()
        ),
    })
    simulation.metadata["methane_storage"] = sizing
    peak_delivery = (float(np.max(delivery)) if hasattr(delivery, "__len__")
                     else float(delivery))
    sizing.update({
        "storage_pressure_bar": plant.methane_storage_pressure_bar,
        "inlet_pressure_bar": plant.co2_outlet_pressure_bar,
        "delivery_pressure_bar": plant.methane_delivery_pressure_bar,
        "compressor_specific_work_kwh_per_kg": compression,
        "expander_specific_work_kwh_per_kg": expansion,
        "expander_reheat_specific_duty_kwh_per_kg": reheat,
        "compressor_capacity_kw": float((output * compression).max()),
        "expander_capacity_kw": peak_delivery * expansion,
        "expander_reheater_capacity_kw": peak_delivery * reheat,
    })
    return sizing


def _electrolyser_sizing_from_perfect_dispatch(
        perfect: SimulationResult, energy: PlantEnergyResult,
        plant: PlantParameters) -> tuple[float, dict[str, float | str]]:
    """Size process electrolysis from the most demanding perfect-information day."""
    h2_per_methane = energy.stoichiometry_kg_per_kg_ch4["H2"]
    hourly_h2 = perfect.hourly["target_methane_kg"] * h2_per_methane
    daily = pd.DataFrame({
        "hydrogen_kg": hourly_h2,
        "operating_hours": (perfect.hourly["target_methane_kg"] > 0).astype(float),
    }).resample("D").sum()
    daily["required_hydrogen_rate_kg_h"] = np.where(
        daily["operating_hours"] > 0,
        daily["hydrogen_kg"] / daily["operating_hours"],
        0.0,
    )
    if daily.empty:
        peak_day, peak_h2, peak_hours, peak_rate = "none", 0.0, 0.0, 0.0
    else:
        peak_stamp = daily["required_hydrogen_rate_kg_h"].idxmax()
        peak = daily.loc[peak_stamp]
        peak_day = peak_stamp.date().isoformat()
        peak_h2 = float(peak["hydrogen_kg"])
        peak_hours = float(peak["operating_hours"])
        peak_rate = float(peak["required_hydrogen_rate_kg_h"])
    capacity_kw = peak_rate * plant.electrolyser_kwh_per_kg_h2
    return capacity_kw, {
        "peak_day": peak_day,
        "peak_daily_hydrogen_kg": peak_h2,
        "peak_day_operating_hours": peak_hours,
        "peak_hydrogen_rate_kg_h": peak_rate,
    }


def _fault_reference_capacities(perfect: SimulationResult,
                                energy: PlantEnergyResult,
                                plant: PlantParameters) -> dict[str, float]:
    """Return fault-only flow references without imposing normal power limits."""
    hourly = perfect.hourly
    if hourly.empty:
        return {
            "process_h2_kg_h": 0.0,
            "storage_h2_charge_kg_h": 0.0,
            "stored_h2_handling_kg_h": 0.0,
            "fuel_cell_h2_kg_h": 0.0,
            "dac_co2_kg_h": 0.0,
        }
    process_electrolyser_kw, _ = _electrolyser_sizing_from_perfect_dispatch(
        perfect, energy, plant,
    )
    process_h2 = process_electrolyser_kw / plant.electrolyser_kwh_per_kg_h2
    storage_h2_charge = float(
        hourly.get("long_h2_charge_kg", pd.Series(0.0, index=hourly.index)).max()
    )
    direct_h2 = hourly.get("direct_h2_kg", pd.Series(0.0, index=hourly.index))
    fuel_cell_h2 = hourly.get(
        "long_fuel_cell_h2_kg", pd.Series(0.0, index=hourly.index),
    )
    handling = float((direct_h2 + fuel_cell_h2).max())
    co2_per_methane = energy.stoichiometry_kg_per_kg_ch4["CO2"]
    fresh_co2 = hourly.get(
        "fresh_co2_kg",
        hourly["methane_kg"] * co2_per_methane
        - hourly.get("direct_co2_kg", pd.Series(0.0, index=hourly.index)),
    )
    storage_co2 = hourly.get(
        "long_co2_charge_kg", pd.Series(0.0, index=hourly.index),
    )
    return {
        "process_h2_kg_h": max(0.0, float(process_h2)),
        "storage_h2_charge_kg_h": max(0.0, storage_h2_charge),
        "stored_h2_handling_kg_h": max(0.0, handling, storage_h2_charge),
        "fuel_cell_h2_kg_h": max(0.0, float(fuel_cell_h2.max())),
        "dac_co2_kg_h": max(0.0, float((fresh_co2 + storage_co2).max())),
    }


def calculate_economics(simulation: SimulationResult, energy: PlantEnergyResult, nominal: float,
                        short_storage: StorageParameters, long_storage: StorageParameters,
                        plant: PlantParameters = PlantParameters(),
                        economic: EconomicParameters = EconomicParameters(),
                        strategy: StrategyConfig = StrategyConfig(),
                        process_electrolyser_capacity_kw: float | None = None,
                        vessel: dict[str, float] | None = None) -> dict[str, Any]:
    """Annualize illustrative CAPEX/OPEX and calculate LCOM."""
    costs = economic.costs if economic.costs is not None else load_costs(economic.costs_path)
    # Recomputed here rather than trusted from dispatch, so it has to be told the same
    # thing about the vessel; otherwise costing would reinstate a buffer the run did
    # not have — or, given a tank specified before the run, silently re-size it to fit
    # the outturn and erase the spill the run actually suffered.
    methane_storage = _attach_methane_storage_profile(
        simulation, energy, plant, buffered=strategy.product_storage, vessel=vessel,
        capacity_factor=strategy.f_socp_long)
    methane_storage_costs = costs["methane_storage"]
    unit_costs = costs["plant_units"]

    def storage_machine_capex(machine: str, capacity_kw: float) -> float:
        item = unit_costs[machine]
        return _scaled_cost(
            item["reference_installed_cost_usd"], capacity_kw,
            item["reference_capacity_kw"], item["scaling_exponent"],
        )

    methane_vessel_capex = (
        methane_storage["capacity_kg"]
        * methane_storage_costs["tank_capex_usd_per_kg"]
    )
    methane_compressor_capex = storage_machine_capex(
        "storage_compressor", methane_storage["compressor_capacity_kw"],
    )
    methane_expander_capex = storage_machine_capex(
        "storage_expander", methane_storage["expander_capacity_kw"],
    )
    # The tank is commissioned holding methane the plant never made, and it can deliver
    # against that stock for months. Left free it is a subsidy in the LCOM denominator,
    # so the opening fill is bought like any other commissioning item.
    methane_initial_fill_capex = (
        methane_storage["initial_inventory_kg"]
        * methane_storage_costs.get("initial_fill_usd_per_kg", 0.0)
    )
    methane_storage_capex = (
        methane_vessel_capex + methane_compressor_capex + methane_expander_capex
        + methane_initial_fill_capex
    )
    methane_storage_opex = (
        methane_storage_capex
        * methane_storage_costs["fixed_opex_fraction_per_year"]
    )
    electrolysis_kw = (
        nominal * energy.breakdown_kwh_per_kg_ch4["electrolysis"]
        if process_electrolyser_capacity_kw is None
        else process_electrolyser_capacity_kw
    )
    compressor_kw = nominal * energy.breakdown_kwh_per_kg_ch4["co2_compression"]
    exchanger_kw = nominal * energy.breakdown_kwh_per_kg_ch4["sabatier_heat_recovered_to_feed"]
    numbered = {"co2_compressor", "sabatier_reactor", "feed_effluent_heat_exchanger"}
    numbered_capacity = {
        "co2_compressor": compressor_kw,
        "sabatier_reactor": nominal,
        "feed_effluent_heat_exchanger": exchanger_kw,
    }
    plant_capex = 0.0
    equipment_capex: dict[str, float] = {
        "methane_storage_vessel": methane_vessel_capex,
        "methane_storage_compressor": methane_compressor_capex,
        "methane_storage_expander": methane_expander_capex,
    }
    for name, item in unit_costs.items():
        if name in {"fuel_cell", "storage_compressor", "storage_expander"}:
            continue
        if name in numbered:
            count = strategy.parallel_reactor_count
            capacity = numbered_capacity[name] / count
        else:
            count = 1
            capacity = electrolysis_kw if name == "electrolyser" else nominal
        reference_capacity = item.get("reference_capacity_kw", item.get("reference_capacity_kg_ch4_h"))
        item_capex = count * _scaled_cost(
            item["reference_installed_cost_usd"], capacity,
            reference_capacity, item["scaling_exponent"],
        )
        equipment_capex[name] = item_capex
        plant_capex += item_capex
    solar_capex = plant.solar_farm_mw * 1000 * costs["solar"]["capex_usd_per_kw"]
    storage_capex = storage_opex = hydrogen_tank_kg = co2_tank_kg = 0.0
    capex_breakdown = {"Solar farm": solar_capex, **{
        name.replace("_", " ").title(): value
        for name, value in equipment_capex.items()
        if not name.startswith(("storage_", "methane_storage_"))
    }, "Product methane buffer": methane_storage_capex}
    opex_breakdown = {
        "Process plant (fixed)": plant_capex * costs["plant_opex"]["fixed_fraction_of_plant_capex_per_year"],
        "Solar farm (fixed)": solar_capex * costs["solar"]["fixed_opex_fraction_per_year"],
        "Product methane buffer (fixed)": methane_storage_opex,
    }
    for storage_label, store in zip(("Short-term storage", "Long-term storage"),
                                    (short_storage, long_storage)):
        previous_capex, previous_opex = storage_capex, storage_opex
        if store.method == "battery":
            power = max(0.0 if math.isinf(store.max_charge_kw) else store.max_charge_kw,
                        0.0 if math.isinf(store.max_discharge_kw) else store.max_discharge_kw)
            capex = store.capacity_kwh * costs["battery"]["energy_capex_usd_per_kwh"] + power * costs["battery"]["power_capex_usd_per_kw"]
            storage_capex += capex
            storage_opex += capex * costs["battery"]["fixed_opex_fraction_per_year"]
        else:
            item = costs["hydrogen_storage"]
            tank = store.capacity_kwh / plant.hydrogen_lhv_kwh_per_kg
            hydrogen_tank_kg += tank
            charge_power = (
                store.hydrogen_charge_capacity_kw
                if store.hydrogen_charge_capacity_kw > 0
                else 0.0 if math.isinf(store.max_charge_kw) else store.max_charge_kw
            )
            discharge_power = (
                store.hydrogen_fuel_cell_capacity_kw
                if store.hydrogen_fuel_cell_capacity_kw > 0
                else 0.0 if math.isinf(store.max_discharge_kw) else store.max_discharge_kw
            )
            fuel_cell = unit_costs["fuel_cell"]
            fuel_cell_capex = _scaled_cost(
                fuel_cell["reference_installed_cost_usd"], discharge_power,
                fuel_cell["reference_capacity_kw"], fuel_cell["scaling_exponent"],
            )
            equipment_capex["storage_fuel_cell"] = (
                equipment_capex.get("storage_fuel_cell", 0.0) + fuel_cell_capex
            )
            h2_compressor_capex = storage_machine_capex(
                "storage_compressor", store.hydrogen_storage_compressor_capacity_kw,
            )
            h2_expander_capex = storage_machine_capex(
                "storage_expander", store.hydrogen_storage_expander_capacity_kw,
            )
            equipment_capex["storage_hydrogen_compressor"] = (
                equipment_capex.get("storage_hydrogen_compressor", 0.0)
                + h2_compressor_capex
            )
            equipment_capex["storage_hydrogen_expander"] = (
                equipment_capex.get("storage_hydrogen_expander", 0.0)
                + h2_expander_capex
            )
            capex = (
                tank * item["tank_capex_usd_per_kg"]
                + charge_power * item["extra_electrolyser_capex_usd_per_kw"]
                + fuel_cell_capex
                + h2_compressor_capex
                + h2_expander_capex
            )
            storage_capex += capex
            storage_opex += capex * item["fixed_opex_fraction_per_year"]
            if store.method == "h2_co2":
                co2_item = costs["co2_storage"]
                co2_tank_kg += store.co2_capacity_kg
                co2_charge_power = store.co2_charge_compressor_capacity_kw
                storage_compressor_capex = storage_machine_capex(
                    "storage_compressor", co2_charge_power,
                )
                storage_expander_capex = storage_machine_capex(
                    "storage_expander", store.co2_storage_expander_capacity_kw,
                )
                equipment_capex["storage_co2_compressor"] = (
                    equipment_capex.get("storage_co2_compressor", 0.0)
                    + storage_compressor_capex
                )
                equipment_capex["storage_co2_expander"] = (
                    equipment_capex.get("storage_co2_expander", 0.0)
                    + storage_expander_capex
                )
                co2_capex = (
                    store.co2_capacity_kg * co2_item["tank_capex_usd_per_kg"]
                    + storage_compressor_capex
                    + storage_expander_capex
                )
                storage_capex += co2_capex
                storage_opex += co2_capex * co2_item["fixed_opex_fraction_per_year"]
        capex_breakdown[storage_label] = storage_capex - previous_capex
        opex_breakdown[f"{storage_label} (fixed)"] = storage_opex - previous_opex
    total_capex = plant_capex + solar_capex + storage_capex + methane_storage_capex
    annual_methane = float(simulation.metrics["average_annual_methane_kg"] or 0.0)
    annual_opex = (plant_capex * costs["plant_opex"]["fixed_fraction_of_plant_capex_per_year"] +
                   solar_capex * costs["solar"]["fixed_opex_fraction_per_year"] + storage_opex +
                   methane_storage_opex +
                   annual_methane * costs["plant_opex"]["variable_usd_per_kg_ch4"])
    opex_breakdown["Production (variable)"] = (
        annual_methane * costs["plant_opex"]["variable_usd_per_kg_ch4"]
    )
    financial = costs["financial"]
    annualized_capex = total_capex * _capital_recovery_factor(financial["real_discount_rate"],
                                                               financial["project_life_years"])
    lcom = (annualized_capex + annual_opex) / annual_methane if annual_methane > 0 else None
    return {"cost_status": costs["metadata"]["status"], "cost_warning": costs["metadata"]["warning"],
            "currency": costs["metadata"]["currency"], "plant_capex_usd": plant_capex,
            "solar_capex_usd": solar_capex, "storage_capex_usd": storage_capex,
            "methane_storage_capex_usd": methane_storage_capex,
            "methane_storage_vessel_capex_usd": methane_vessel_capex,
            "methane_initial_fill_capex_usd": methane_initial_fill_capex,
            "methane_storage_compressor_capex_usd": methane_compressor_capex,
            "methane_storage_expander_capex_usd": methane_expander_capex,
            "methane_storage_fixed_opex_usd_per_year": methane_storage_opex,
            "methane_storage_capacity_kg": methane_storage["capacity_kg"],
            "total_capex_usd": total_capex, "annualized_capex_usd_per_year": annualized_capex,
            "annual_opex_usd_per_year": annual_opex, "average_annual_methane_kg": annual_methane,
            "lcom_usd_per_kg_ch4": lcom, "hydrogen_tank_capacity_kg": hydrogen_tank_kg,
            "co2_tank_capacity_kg": co2_tank_kg,
            "equipment_capex_usd": equipment_capex,
            "capex_breakdown_usd": capex_breakdown,
            "opex_breakdown_usd_per_year": opex_breakdown}


def _configured_store(template: StorageParameters, capacity: float, initial_soc: float,
                      capacity_fraction: float, *,
                      co2_capacity_kg: float = 0.0) -> tuple[StorageParameters, float]:
    """Install selected energy capacity with unconstrained internal transfer power."""
    actual_capacity = max(0.0, capacity * capacity_fraction)
    energy_scale = actual_capacity / capacity if capacity > 0 else 1.0
    configured = replace(
        template,
        capacity_kwh=actual_capacity,
        co2_capacity_kg=max(0.0, co2_capacity_kg * capacity_fraction),
        max_charge_kw=math.inf,
        max_discharge_kw=math.inf,
    )
    return configured, min(actual_capacity, initial_soc * energy_scale)


def build_equipment_sizing(
        energy: PlantEnergyResult, sizing: StorageSizingResult,
        short: StorageParameters, long: StorageParameters,
        plant: PlantParameters, strategy: StrategyConfig,
        perfect: SimulationResult) -> tuple[dict[str, Any], ...]:
    """Build the user-facing required and installed equipment sizing register."""
    nominal = sizing.nominal_methane_kg_h
    count = strategy.parallel_reactor_count
    electrolysis_kw, electrolyser_basis = _electrolyser_sizing_from_perfect_dispatch(
        perfect, energy, plant,
    )
    compressor_kw = nominal * energy.breakdown_kwh_per_kg_ch4["co2_compression"]
    exchanger_kw = nominal * energy.breakdown_kwh_per_kg_ch4["sabatier_heat_recovered_to_feed"]
    rows: list[dict[str, Any]] = []

    def add(category: str, unit_name: str, number: int, required: float,
            installed: float, unit: str, basis: str) -> None:
        rows.append({
            "category": category, "unit_name": unit_name, "count": number,
            "required_size_per_unit": required / number if number else 0.0,
            "installed_size_per_unit": installed / number if number else 0.0,
            "required_total_size": required, "installed_total_size": installed,
            "unit": unit, "sizing_basis": basis,
        })

    add("Generation", "Solar farm", 1, plant.solar_farm_mw, plant.solar_farm_mw, "MWp",
        "The installed photovoltaic nameplate capacity is entered directly by the user on the Single site tab and is not calculated by dispatch.")
    add("Common process", "Process electrolyser", 1, electrolysis_kw, electrolysis_kw, "kW",
        "The perfect-information schedule calculates hydrogen demand for every UTC day; "
        f"the limiting day {electrolyser_basis['peak_day']} requires "
        f"{electrolyser_basis['peak_daily_hydrogen_kg']:,.3g} kg H2 over "
        f"{electrolyser_basis['peak_day_operating_hours']:,.3g} operating hours, so "
        f"{electrolyser_basis['peak_hydrogen_rate_kg_h']:,.3g} kg H2/h at "
        f"{plant.electrolyser_kwh_per_kg_h2:,.3g} kWh/kg H2 sets this rating and ensures every planned day can be supplied.")
    add("Common process", "DAC fan and contactor", 1, nominal, nominal, "kg CH4/h equivalent",
        "Sized for the CO2 capture flow stoichiometrically required at the full calculated methane nameplate output; the displayed unit is its equivalent methane throughput.")
    add("Common process", "Carbonator", 1, nominal, nominal, "kg CH4/h equivalent",
        "One common carbonator is rated for the stoichiometric solids and CO2 flow at full methane nameplate output and turns down to active trains divided by installed trains.")
    add("Common process", "Calciner", 1, nominal, nominal, "kg CH4/h equivalent",
        "One common calciner is rated for the stoichiometric solids and CO2 flow at full methane nameplate output and turns down to active trains divided by installed trains.")
    add("Parallel trains", "CO2 compressor", count, compressor_kw, compressor_kw, "kW",
        "The full-nameplate CO2 mass flow is compressed from the calculated calcination equilibrium pressure to the specified outlet pressure; total duty is divided equally across one compressor per train.")
    add("Parallel trains", "Sabatier reactor", count, nominal, nominal, "kg CH4/h",
        "The calculated maximum methane nameplate output is divided equally across the user-selected train count; every active reactor therefore operates at exactly its displayed 100% load.")
    add("Parallel trains", "Feed-effluent heat exchanger", count, exchanger_kw, exchanger_kw, "kW",
        "Recovered Sabatier heat first supplies the full-nameplate CO2 and H2 feed-heating duty; that recovered duty is divided equally across one exchanger per train.")

    for label, store, required_capacity, charge_column, discharge_column in (
        ("Short", short, sizing.short_capacity_kwh,
         "short_charge_kwh", "short_discharge_kwh"),
        ("Long", long, sizing.long_capacity_kwh,
         "long_charge_kwh", "long_discharge_kwh"),
    ):
        required_charge = (
            float(perfect.hourly[charge_column].max()) if len(perfect.hourly) else 0.0
        )
        required_discharge = (
            float(perfect.hourly[discharge_column].max()) if len(perfect.hourly) else 0.0
        )
        carrier = "battery energy" if store.method == "battery" else "hydrogen LHV"
        energy_basis = (
            f"The perfect-information within-day SOC range gives the required {carrier} "
            "capacity; short-term storage is installed at exactly 100% of that requirement."
            if label == "Short" else
            f"The perfect-information longer residual SOC range gives the required {carrier} "
            "capacity and f_SOCP sets the installed long-term value."
        )
        add("Storage", f"{label} storage energy", 1, required_capacity,
            store.capacity_kwh, "kWh carrier", energy_basis)
        installed_charge = 0.0 if math.isinf(store.max_charge_kw) else store.max_charge_kw
        installed_discharge = 0.0 if math.isinf(store.max_discharge_kw) else store.max_discharge_kw
        add("Storage", f"{label} storage charge power", 1, required_charge,
            installed_charge, "kW", "The maximum one-hour charge transfer observed after the unconstrained perfect-information dispatch is reported as the required rating; no transfer-power limit is imposed during dispatch.")
        add("Storage", f"{label} storage discharge power", 1, required_discharge,
            installed_discharge, "kW", "The maximum one-hour discharge transfer observed after the unconstrained perfect-information dispatch is reported as the required rating; no transfer-power limit is imposed during dispatch.")
        if store.method in {"hydrogen", "h2_co2"}:
            required_kg = required_capacity / plant.hydrogen_lhv_kwh_per_kg
            installed_kg = store.capacity_kwh / plant.hydrogen_lhv_kwh_per_kg
            add("Hydrogen storage", f"{label} hydrogen tank", 1, required_kg,
                installed_kg, "kg H2", "The required and installed hydrogen carrier-energy capacities are divided by the editable hydrogen lower heating value to obtain tank inventory in kilograms.")
            electrolyser_charge = (
                float(perfect.hourly["long_h2_charge_kg"].max())
                * plant.electrolyser_kwh_per_kg_h2
                if label == "Long" and len(perfect.hourly)
                else required_charge
            )
            add("Hydrogen storage", f"{label} storage electrolyser", 1, electrolyser_charge,
                electrolyser_charge, "kW", "Sized from the maximum perfect-information hydrogen charging rate after direct process demand is served.")
            h2_compressor_power = (
                float(perfect.hourly["long_h2_charge_kg"].max())
                * store.hydrogen_compression_kwh_per_kg
                if label == "Long" and len(perfect.hourly) else 0.0
            )
            h2_expander_power = (
                float((perfect.hourly["long_direct_h2_kg"]
                       + perfect.hourly["long_fuel_cell_h2_kg"]).max())
                * store.hydrogen_expansion_kwh_per_kg
                if label == "Long" and len(perfect.hourly) else 0.0
            )
            add("Hydrogen storage", f"{label} hydrogen storage compressor", 1,
                h2_compressor_power, h2_compressor_power, "kW",
                f"Sized from peak perfect-information H2 vessel inflow and {store.hydrogen_compression_kwh_per_kg:.3g} kWh/kg staged compression from {plant.co2_outlet_pressure_bar:g} to {plant.hydrogen_storage_pressure_bar:g} bar.")
            add("Hydrogen storage", f"{label} hydrogen storage expander", 1,
                h2_expander_power, h2_expander_power, "kW",
                f"Sized from peak perfect-information H2 vessel outflow and {store.hydrogen_expansion_kwh_per_kg:.3g} kWh/kg staged expansion from {plant.hydrogen_storage_pressure_bar:g} to {plant.co2_outlet_pressure_bar:g} bar.")
            h2_reheater_power = (
                float((perfect.hourly["long_direct_h2_kg"]
                       + perfect.hourly["long_fuel_cell_h2_kg"]).max())
                * store.hydrogen_expansion_reheat_kwh_per_kg
                if label == "Long" and len(perfect.hourly) else 0.0
            )
            add("Hydrogen storage", f"{label} hydrogen expander reheater", 1,
                h2_reheater_power, h2_reheater_power, "kW",
                f"Electrical interstage reheating at COP 1 returning H2 to {plant.intercool_temperature_k:.6g} K between each of the {plant.storage_machine_stages:d} expansion stages, at {store.hydrogen_expansion_reheat_kwh_per_kg:.3g} kWh/kg H2 discharged. Hydrogen sits above its Joule-Thomson inversion temperature, so a throttle valve would instead reject this enthalpy as waste heat; the expander converts it to net bus work.")
            fuel_cell_power = (
                float(perfect.hourly["long_fuel_cell_h2_kg"].max())
                * store.hydrogen_fuel_cell_kwh_per_kg
                if label == "Long" and len(perfect.hourly) else 0.0
            )
            add("Hydrogen storage", f"{label} fuel cell", 1, fuel_cell_power,
                fuel_cell_power, "kW", "Sized from the peak perfect-information hydrogen flow converted back to bus electricity after stored hydrogen has first been reserved for Sabatier feed.")
            if store.method == "h2_co2" and label == "Long":
                required_co2 = sizing.long_co2_capacity_kg
                co2_compressor_power = (
                    float(perfect.hourly["long_co2_charge_kg"].max())
                    * store.co2_compression_kwh_per_kg
                    if len(perfect.hourly) else 0.0
                )
                add("H2 + CO2 gas storage", "Long CO2 storage vessel", 1,
                    required_co2, store.co2_capacity_kg, "kg CO2",
                    "The perfect-information hydrogen requirement is converted to its Sabatier-stoichiometric CO2 inventory; f_SOCP scales both gas vessels by the same fraction before the capacities pass unchanged to imperfect dispatch.")
                add("H2 + CO2 gas storage", "CO2 storage charging compressor", 1,
                    co2_compressor_power, co2_compressor_power, "kW",
                    f"Sized from peak perfect-information CO2 vessel inflow and {store.co2_compression_kwh_per_kg:.3g} kWh/kg staged compression from {plant.co2_outlet_pressure_bar:g} to {plant.co2_storage_pressure_bar:g} bar.")
                co2_expander_power = (
                    float(perfect.hourly["long_direct_co2_kg"].max())
                    * store.co2_expansion_kwh_per_kg
                    if len(perfect.hourly) else 0.0
                )
                add("H2 + CO2 gas storage", "CO2 storage discharge expander", 1,
                    co2_expander_power, co2_expander_power, "kW",
                    f"Sized from peak perfect-information CO2 vessel outflow and {store.co2_expansion_kwh_per_kg:.3g} kWh/kg staged expansion from {plant.co2_storage_pressure_bar:g} to {plant.co2_outlet_pressure_bar:g} bar.")
                co2_reheater_power = (
                    float(perfect.hourly["long_direct_co2_kg"].max())
                    * store.co2_expansion_reheat_kwh_per_kg
                    if len(perfect.hourly) else 0.0
                )
                add("H2 + CO2 gas storage", "CO2 expander reheater", 1,
                    co2_reheater_power, co2_reheater_power, "kW",
                    f"Electrical interstage reheating at COP 1 returning CO2 to {plant.intercool_temperature_k:.6g} K between each of the {plant.storage_machine_stages:d} expansion stages, at {store.co2_expansion_reheat_kwh_per_kg:.3g} kWh/kg CO2 discharged. Most of this duty is the unavoidable isothermal letdown enthalpy that a plain throttle valve would also require; only the balance pays for the expander's recovered work.")
    methane_storage = perfect.metadata["methane_storage"]
    add(
        "Product handling", "Product methane buffer vessel", 1,
        methane_storage["capacity_kg"], methane_storage["capacity_kg"], "kg CH4",
        "Sized from the range of the cumulative hourly difference between methane "
        "production and continuous delivery at the mean production rate; the vessel "
        "inventory is cyclic over the complete evaluation period.",
    )
    add(
        "Product handling", "Product methane compressor", 1,
        methane_storage["compressor_capacity_kw"], methane_storage["compressor_capacity_kw"],
        "kW", f"Sized from peak hourly methane production and {methane_storage['compressor_specific_work_kwh_per_kg']:.3g} kWh/kg staged compression from {methane_storage['inlet_pressure_bar']:g} to {methane_storage['storage_pressure_bar']:g} bar.",
    )
    add(
        "Product handling", "Product methane expander", 1,
        methane_storage["expander_capacity_kw"], methane_storage["expander_capacity_kw"],
        "kW", f"Sized from constant methane delivery and {methane_storage['expander_specific_work_kwh_per_kg']:.3g} kWh/kg staged expansion from {methane_storage['storage_pressure_bar']:g} to {methane_storage['delivery_pressure_bar']:g} bar.",
    )
    add(
        "Product handling", "Product methane expander reheater", 1,
        methane_storage["expander_reheater_capacity_kw"],
        methane_storage["expander_reheater_capacity_kw"],
        "kW", f"Electrical interstage reheating at COP 1 returning the gas to {plant.intercool_temperature_k:.6g} K between each of the {plant.storage_machine_stages:d} expansion stages, at {methane_storage['expander_reheat_specific_duty_kwh_per_kg']:.3g} kWh/kg of delivered methane. Most of this duty is the unavoidable isothermal letdown enthalpy that a plain throttle valve would also require, and without it the expander would discharge into the two-phase region.",
    )
    return tuple(rows)


def run_case(actual: pd.DataFrame, forecast: pd.DataFrame | None, *,
             plant: PlantParameters = PlantParameters(), thermal: ThermalParameters = ThermalParameters(),
             strategy: StrategyConfig = StrategyConfig(),
             short_storage_template: StorageParameters = StorageParameters(),
             long_storage_template: StorageParameters = StorageParameters(self_discharge_fraction_per_h=1e-5),
             economic: EconomicParameters = EconomicParameters(), faults: Sequence[FaultEvent] | None = None,
             fault_scenario: FaultScenario | None = None,
             include_faulted: bool | None = None,
             save_outputs: bool = False, output_root: str | Path = DEFAULT_OUTPUT_DIR,
             metadata: dict[str, Any] | None = None,
             include_imperfect: bool = True,
             include_baseline: bool = False,
             use_forecast_commitment: bool = False,
             sizing_profile: pd.DataFrame | None = None,
             progress: ProgressCallback | None = None) -> CaseResult:
    """Run sizing and dispatch using one shared forecast commitment schedule.

    ``sizing_profile`` is the weather a designer would have had before building:
    give it the training period and the forecast-driven cases run a plant sized on
    that record rather than on the decade they are evaluated over. Omit it to size
    every case on ``actual``.
    """
    if faults is None:
        faults = (generate_fault_events(actual.index, fault_scenario)
                  if fault_scenario is not None else ())
    faults = tuple(faults)
    if include_faulted is None:
        include_faulted = bool(faults) or fault_scenario is not None
    if include_faulted and not include_imperfect:
        raise ValueError("include_faulted requires include_imperfect")

    def configured_hydrogen_conversion(
            store: StorageParameters, energy_result: PlantEnergyResult) -> StorageParameters:
        if store.method not in {"hydrogen", "h2_co2"}:
            return store
        values: dict[str, float] = {
            "hydrogen_lhv_kwh_per_kg": plant.hydrogen_lhv_kwh_per_kg,
            "hydrogen_electrolyser_kwh_per_kg": plant.electrolyser_kwh_per_kg_h2,
            "hydrogen_fuel_cell_kwh_per_kg": plant.fuel_cell_kwh_per_kg_h2,
            "hydrogen_compression_kwh_per_kg": (
                energy_result.gas_storage_work_kwh_per_kg["hydrogen_compression"]
            ),
            "hydrogen_expansion_kwh_per_kg": (
                energy_result.gas_storage_work_kwh_per_kg["hydrogen_expansion"]
            ),
            "hydrogen_expansion_reheat_kwh_per_kg": (
                energy_result.gas_storage_work_kwh_per_kg["hydrogen_expansion_reheat"]
            ),
        }
        if store.method == "h2_co2":
            h2_per_ch4 = energy_result.stoichiometry_kg_per_kg_ch4["H2"]
            co2_per_ch4 = energy_result.stoichiometry_kg_per_kg_ch4["CO2"]
            upstream_keys = (
                "dac_fan", "co2_compression", "calcination_reaction_heat",
                "unrecovered_solids_sensible_heat", "carbonator_air_electric_heat",
            )
            values.update({
                "co2_to_hydrogen_mass_ratio": co2_per_ch4 / h2_per_ch4,
                "co2_production_kwh_per_kg": (
                    sum(energy_result.breakdown_kwh_per_kg_ch4[key] for key in upstream_keys)
                    / co2_per_ch4
                ),
                "co2_compression_kwh_per_kg": (
                    energy_result.gas_storage_work_kwh_per_kg["co2_compression"]
                ),
                "co2_expansion_kwh_per_kg": (
                    energy_result.gas_storage_work_kwh_per_kg["co2_expansion"]
                ),
                "co2_expansion_reheat_kwh_per_kg": (
                    energy_result.gas_storage_work_kwh_per_kg["co2_expansion_reheat"]
                ),
            })
        return replace(
            store,
            **values,
        )

    _report_progress(progress, "Plant: calculating stoichiometry and E_req.")
    energy = calculate_plant_energy(plant)
    short_storage_template = configured_hydrogen_conversion(short_storage_template, energy)
    long_storage_template = configured_hydrogen_conversion(long_storage_template, energy)
    _report_progress(progress, "Storage: sizing cyclic energy capacity.")
    sizing = size_perfect_storage(actual, energy, plant, thermal, short_storage_template,
                                  long_storage_template, strategy, progress)
    short, short_initial = _configured_store(
        short_storage_template, sizing.short_capacity_kwh,
        sizing.short_initial_soc_kwh, 1.0,
    )
    # f_SOCP is the fraction of the required seasonal storage actually built, and a
    # plant whose seasonal store is its product vessel builds no upstream store at all;
    # f_SOCP then sizes the tank instead. Building both would be two seasonal stores in
    # series, each sized against the other's leftovers.
    energy_store_factor = 0.0 if strategy.product_storage else strategy.f_socp_long
    long, long_initial = _configured_store(
        long_storage_template, sizing.long_capacity_kwh,
        sizing.long_initial_soc_kwh, energy_store_factor,
        co2_capacity_kg=sizing.long_co2_capacity_kg,
    )
    # A plant is sized once, before it is built, from the years its designer had.
    # With sizing_profile given, the forecast-driven cases run that plant: capacities,
    # throughput and equipment ratings all fixed by the earlier period, and the
    # evaluation decade is simply what it then has to live through. Perfect
    # information keeps its own plant, sized on the decade it actually meets, so the
    # comparison carries the cost of sizing blind as well as of dispatching blind.
    # Leaving sizing_profile None sizes every case on the evaluation weather, which is
    # what the model did before and what the tests for a single plant still expect.
    historic_sizing, historic_reference, historic_vessel = sizing, None, None
    historic_short, historic_short_initial = short, short_initial
    historic_long, historic_long_initial = long, long_initial
    if sizing_profile is not None:
        _report_progress(progress, "Storage: sizing the plant on the build-year record.")
        historic_sizing = size_perfect_storage(
            sizing_profile, energy, plant, thermal, short_storage_template,
            long_storage_template, strategy, progress,
        )
        historic_short, historic_short_initial = _configured_store(
            short_storage_template, historic_sizing.short_capacity_kwh,
            historic_sizing.short_initial_soc_kwh, 1.0,
        )
        historic_long, historic_long_initial = _configured_store(
            long_storage_template, historic_sizing.long_capacity_kwh,
            historic_sizing.long_initial_soc_kwh, energy_store_factor,
            co2_capacity_kg=historic_sizing.long_co2_capacity_kg,
        )
        # Equipment ratings follow the same rule as capacities: whatever the design
        # study over the build-year record called for. Rating this plant from peaks
        # observed in the decade after it was built would be the clairvoyance the
        # split exists to remove.
        _report_progress(progress, "Dispatch: running the build-year design reference.")
        historic_reference = _simulate_dispatch_with_methane_storage(
            sizing_profile, sizing_profile, energy,
            historic_sizing.nominal_methane_kg_h, historic_short, historic_long,
            plant, thermal, strategy, (),
            short_initial_soc_kwh=historic_short_initial,
            long_initial_soc_kwh=historic_long_initial,
            progress=progress, progress_label="Design-reference dispatch progress",
        )
        # The tank is specified by the same design study as the rest of the plant.
        historic_vessel = dict(historic_reference.metadata["methane_storage"])
    shared_commitment = None
    historic_commitment = None
    if include_imperfect or use_forecast_commitment:
        if forecast is None:
            raise ValueError(
                "forecast is required when include_imperfect or use_forecast_commitment is true"
            )

        def commitment_for(nominal, short_store, long_store, short_soc, long_soc):
            """Commit reactors from the forecast, for one particular plant.

            Each plant plans against its own throughput and its own stores. The
            perfect-information case keeps the schedule it always had, so splitting
            the plants leaves it exactly where it was.
            """
            schedule = build_target_schedule(
                forecast, nominal, energy, plant, thermal, strategy,
            ).copy()
            if strategy.reactor_scheduling_mode != "seasonal":
                planning_reference = simulate_dispatch(
                    actual, forecast, energy, nominal, short_store, long_store,
                    plant, thermal, strategy, (),
                    short_initial_soc_kwh=short_soc,
                    long_initial_soc_kwh=long_soc, progress=progress,
                    progress_label="Commitment planning progress",
                )
                schedule["planned_reactor_count"] = (
                    planning_reference.hourly["planned_reactor_count"].round().astype(int)
                )
            schedule["daily_reactor_count"] = (
                schedule["planned_reactor_count"].groupby(
                    schedule.index.normalize()
                ).transform("max").astype(int)
            )
            return schedule

        _report_progress(progress, "Dispatch: constructing the shared forecast commitment schedule.")
        shared_commitment = commitment_for(
            sizing.nominal_methane_kg_h, short, long, short_initial, long_initial,
        )
        historic_commitment = shared_commitment
        if historic_reference is not None:
            _report_progress(
                progress, "Dispatch: constructing the built plant's commitment schedule.")
            historic_commitment = commitment_for(
                historic_sizing.nominal_methane_kg_h, historic_short, historic_long,
                historic_short_initial, historic_long_initial,
            )
    _report_progress(progress, "Dispatch: running the perfect-information case.")
    perfect = _simulate_dispatch_with_methane_storage(actual, actual, energy, sizing.nominal_methane_kg_h, short, long,
                                plant, thermal, strategy, (), short_initial_soc_kwh=short_initial,
                                long_initial_soc_kwh=long_initial, progress=progress,
                                progress_label="Perfect dispatch progress",
                                commitment_schedule=shared_commitment)
    imperfect = None
    imperfect_with_faults = None
    if include_imperfect:
        _report_progress(progress, "Dispatch: running the climatology-forecast case.")
        imperfect = _simulate_dispatch_with_methane_storage(
            actual, forecast, energy, historic_sizing.nominal_methane_kg_h,
            historic_short, historic_long,
            plant, thermal, strategy, (),
            short_initial_soc_kwh=historic_short_initial,
            long_initial_soc_kwh=historic_long_initial, progress=progress,
            progress_label="Imperfect dispatch progress",
            commitment_schedule=historic_commitment, vessel=historic_vessel)
        if include_faulted:
            _report_progress(progress, "Dispatch: running the imperfect-with-faults case.")
            # Faulting is a bolt-on to the imperfect case, so the derating reference is
            # the plant that is being faulted, not the perfect-information one.
            fault_capacities = _fault_reference_capacities(imperfect, energy, plant)
            imperfect_with_faults = _simulate_dispatch_with_methane_storage(
                actual, forecast, energy, historic_sizing.nominal_methane_kg_h,
                historic_short, historic_long,
                plant, thermal, strategy, faults,
                short_initial_soc_kwh=historic_short_initial,
                long_initial_soc_kwh=historic_long_initial,
                progress=progress,
                progress_label="Faulted dispatch progress",
                commitment_schedule=historic_commitment,
                fault_capacities=fault_capacities, vessel=historic_vessel,
            )
    else:
        _report_progress(progress, "Dispatch: perfect-information-only mode; forecast case skipped.")

    baseline = None
    baseline_stores = None
    if include_baseline:
        # The "no autonomy" reference the brief asks to compare against: the same
        # sized plant with no storage and no scheduling, run flat out whenever the
        # sun allows and curtailing whatever it cannot absorb. It carries the same
        # equipment faults as the faulted case, because breakdowns do not care how
        # the plant is dispatched; comparing a faulted plant against a fault-free
        # reference would credit storage with reliability it does not provide.
        _report_progress(progress, "Dispatch: running the no-storage baseline.")
        empty_store = replace(
            short_storage_template, capacity_kwh=0.0, co2_capacity_kg=0.0,
            self_discharge_fraction_per_h=0.0, co2_self_discharge_fraction_per_h=0.0,
            initial_soc_fraction=0.0,
        )
        baseline_strategy = replace(
            strategy, short_strategy="hard_shutdown", f_socp_long=0.0,
            reactor_scheduling_mode="daily_storage_aware",
        )
        flat_out = build_target_schedule(
            actual, historic_sizing.nominal_methane_kg_h, energy, plant, thermal,
            baseline_strategy,
        )
        # Commit every train whenever there is daylight; no storage-aware rounding.
        flat_out["planned_reactor_count"] = np.where(
            flat_out["daylight"], strategy.parallel_reactor_count, 0,
        ).astype(int)
        flat_out["daily_reactor_count"] = flat_out["planned_reactor_count"]
        baseline_faults = faults if include_faulted else ()
        baseline = _simulate_dispatch_with_methane_storage(
            actual, actual, energy, historic_sizing.nominal_methane_kg_h,
            empty_store, empty_store, plant, thermal, baseline_strategy,
            baseline_faults,
            short_initial_soc_kwh=0.0, long_initial_soc_kwh=0.0,
            progress=progress, progress_label="Baseline dispatch progress",
            commitment_schedule=flat_out,
            fault_capacities=_fault_reference_capacities(
                imperfect if imperfect is not None else perfect, energy, plant)
                             if baseline_faults else None,
        )
        baseline_stores = (empty_store, empty_store)

    def observed_power(simulation: SimulationResult) -> dict[str, Any]:
        def peak(column: str) -> float:
            return float(simulation.hourly[column].max()) if len(simulation.hourly) else 0.0

        values = {
            "power_limits_applied": False,
            "short_required_charge_kw": peak("short_charge_kwh"),
            "short_required_discharge_kw": peak("short_discharge_kwh"),
            "long_required_charge_kw": peak("long_charge_kwh"),
            "long_required_discharge_kw": peak("long_discharge_kwh"),
            "long_hydrogen_charge_kw": (
                peak("long_h2_charge_kg") * plant.electrolyser_kwh_per_kg_h2
            ),
            "long_hydrogen_fuel_cell_kw": (
                peak("long_fuel_cell_h2_kg") * plant.fuel_cell_kwh_per_kg_h2
            ),
            "long_hydrogen_storage_compressor_kw": (
                peak("long_h2_charge_kg") * long.hydrogen_compression_kwh_per_kg
            ),
            "long_hydrogen_storage_expander_kw": (
                float((simulation.hourly["long_direct_h2_kg"]
                       + simulation.hourly["long_fuel_cell_h2_kg"]).max())
                if len(simulation.hourly) else 0.0
            ) * long.hydrogen_expansion_kwh_per_kg,
            "long_co2_charge_kg_h": peak("long_co2_charge_kg"),
            "long_co2_compressor_charge_kw": (
                peak("long_co2_charge_kg") * long.co2_compression_kwh_per_kg
            ),
            "long_co2_storage_expander_kw": (
                peak("long_direct_co2_kg") * long.co2_expansion_kwh_per_kg
            ),
        }
        # Retain these keys for saved-output compatibility. They mean the rating
        # that would be needed to reproduce this unconstrained dispatch, not a
        # limit imposed during the run.
        values.update({
            "short_installed_charge_kw": values["short_required_charge_kw"],
            "short_installed_discharge_kw": values["short_required_discharge_kw"],
            "long_installed_charge_kw": values["long_required_charge_kw"],
            "long_installed_discharge_kw": values["long_required_discharge_kw"],
        })
        return values

    perfect_power = observed_power(perfect)
    perfect.metadata["storage_power"] = perfect_power
    if imperfect is not None:
        imperfect.metadata["storage_power"] = observed_power(imperfect)
    if imperfect_with_faults is not None:
        imperfect_with_faults.metadata["storage_power"] = observed_power(imperfect_with_faults)
    sizing = replace(
        sizing,
        short_required_charge_power_kw=perfect_power["short_required_charge_kw"],
        short_required_discharge_power_kw=perfect_power["short_required_discharge_kw"],
        long_required_charge_power_kw=perfect_power["long_required_charge_kw"],
        long_required_discharge_power_kw=perfect_power["long_required_discharge_kw"],
    )
    perfect.warnings.extend(item for item in sizing.warnings if item not in perfect.warnings)
    # The forecast-driven cases inherit their own plant's sizing warnings, which are
    # not the perfect plant's once the two are sized from different records.
    historic_warnings = list(historic_sizing.warnings)
    realised_deficit = 0.0
    if historic_reference is not None:
        realised_deficit = realised_cyclic_deficit(
            actual, historic_sizing.nominal_methane_kg_h, energy, plant, thermal,
            strategy, historic_short, historic_long,
        )
        if realised_deficit > 1e-6:
            historic_warnings.append(
                "The plant sized on the build-year record runs an average cyclic "
                f"energy deficit of {realised_deficit:,.0f} kWh/year on the weather it "
                "was evaluated against, so its storage was under-built for the period "
                "that followed. Dispatch continues and reports the production "
                "shortfall."
            )
    for simulation in (imperfect, imperfect_with_faults):
        if simulation is not None:
            simulation.warnings.extend(
                item for item in historic_warnings if item not in simulation.warnings
            )
    process_electrolyser_kw, _ = _electrolyser_sizing_from_perfect_dispatch(
        perfect, energy, plant,
    )

    def costed_stores(short_store, long_store, power):
        """Rate a plant's machinery from the dispatch its design study was based on.

        Power-transfer CAPEX uses the peak rating observed in that dispatch, while
        the dispatch itself remains unconstrained by the rating.
        """
        return (
            replace(
                short_store,
                max_charge_kw=power["short_required_charge_kw"],
                max_discharge_kw=power["short_required_discharge_kw"],
            ),
            replace(
                long_store,
                max_charge_kw=power["long_required_charge_kw"],
                max_discharge_kw=power["long_required_discharge_kw"],
                hydrogen_charge_capacity_kw=power["long_hydrogen_charge_kw"],
                hydrogen_fuel_cell_capacity_kw=power["long_hydrogen_fuel_cell_kw"],
                hydrogen_storage_compressor_capacity_kw=(
                    power["long_hydrogen_storage_compressor_kw"]
                ),
                hydrogen_storage_expander_capacity_kw=(
                    power["long_hydrogen_storage_expander_kw"]
                ),
                co2_charge_compressor_capacity_kw=power["long_co2_compressor_charge_kw"],
                co2_storage_expander_capacity_kw=power["long_co2_storage_expander_kw"],
            ),
        )

    costed_short, costed_long = costed_stores(short, long, perfect_power)
    # The plant sized before it was built is costed on its own design study, so its
    # CAPEX differs from the perfect-information plant's. That difference — capital
    # committed against the wrong decade — is what the comparison is for.
    historic_electrolyser_kw = process_electrolyser_kw
    historic_costed_short, historic_costed_long = costed_short, costed_long
    if historic_reference is not None:
        historic_power = observed_power(historic_reference)
        historic_costed_short, historic_costed_long = costed_stores(
            historic_short, historic_long, historic_power,
        )
        historic_electrolyser_kw, _ = _electrolyser_sizing_from_perfect_dispatch(
            historic_reference, energy, plant,
        )
    _report_progress(progress, "Economics: calculating CAPEX, OPEX, and LCOM.")
    ep = calculate_economics(
        perfect, energy, sizing.nominal_methane_kg_h, costed_short, costed_long,
        plant, economic, strategy,
        process_electrolyser_kw,
    )
    ei = None
    eif = None
    if imperfect is not None:
        ei = calculate_economics(
            imperfect, energy, historic_sizing.nominal_methane_kg_h,
            historic_costed_short, historic_costed_long,
            plant, economic, strategy,
            historic_electrolyser_kw, vessel=historic_vessel,
        )
        ei["relative_prediction_cost_ratio"] = (ei["lcom_usd_per_kg_ch4"] / ep["lcom_usd_per_kg_ch4"]
                                                 if ei["lcom_usd_per_kg_ch4"] is not None and ep["lcom_usd_per_kg_ch4"] else None)
    if imperfect_with_faults is not None:
        eif = calculate_economics(
            imperfect_with_faults, energy, historic_sizing.nominal_methane_kg_h,
            historic_costed_short, historic_costed_long, plant, economic, strategy,
            historic_electrolyser_kw, vessel=historic_vessel,
        )

        def ratio(numerator: float | None, denominator: float | None) -> float | None:
            return numerator / denominator if numerator is not None and denominator else None

        eif["relative_fault_cost_ratio"] = ratio(
            eif["lcom_usd_per_kg_ch4"],
            ei["lcom_usd_per_kg_ch4"] if ei is not None else None,
        )
        eif["relative_combined_cost_ratio"] = ratio(
            eif["lcom_usd_per_kg_ch4"], ep["lcom_usd_per_kg_ch4"],
        )
        baseline_methane = imperfect.metrics["methane_total_kg"] if imperfect is not None else 0.0
        faulted_methane = imperfect_with_faults.metrics["methane_total_kg"]
        perfect_methane = perfect.metrics["methane_total_kg"]
        imperfect_with_faults.metrics.update({
            "incremental_fault_methane_loss_kg": baseline_methane - faulted_methane,
            "incremental_fault_curtailment_change_kwh": (
                imperfect_with_faults.metrics["curtailed_energy_kwh"]
                - imperfect.metrics["curtailed_energy_kwh"]
            ),
            "relative_fault_production_ratio": ratio(faulted_methane, baseline_methane),
            "relative_combined_production_ratio": ratio(faulted_methane, perfect_methane),
        })
        hourly_loss = (
            imperfect.hourly["methane_kg"]
            - imperfect_with_faults.hourly["methane_kg"]
        )
        imperfect_with_faults.metrics["incremental_fault_loss_hours"] = int(
            (hourly_loss > 1e-9).sum()
        )
        for component in ("battery", "hydrogen", "sabatier", "dac"):
            active = imperfect_with_faults.hourly[f"{component}_fault_active"] > 0.5
            derated = imperfect_with_faults.hourly[f"{component}_capacity_fraction"] < 1.0 - 1e-12
            imperfect_with_faults.metrics[f"{component}_binding_loss_hours"] = int(
                (active & derated & (hourly_loss > 1e-9)).sum()
            )
            imperfect_with_faults.metrics[f"{component}_event_count"] = sum(
                event.component == component for event in faults
            )
        if fault_scenario is not None:
            imperfect_with_faults.metadata["fault_scenario"] = asdict(fault_scenario)
    raw_fault_events = [_fault_event_dict(event) for event in faults]
    event_schedule_digest = sha256(
        json.dumps(raw_fault_events, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    payload = {"plant": asdict(plant), "thermal": asdict(thermal), "strategy": asdict(strategy),
               "short_storage": asdict(short), "long_storage": asdict(long),
               "economic": (economic.costs if economic.costs is not None
                            else {"costs_path": economic.costs_path}),
               "start": str(actual.index.min()), "end": str(actual.index.max()),
                "include_imperfect": include_imperfect,
                "include_faulted": include_faulted,
                "use_forecast_commitment": use_forecast_commitment,
                "weather_digest": sha256(actual["capacity_factor"].to_numpy().tobytes()).hexdigest()[:16],
                "forecast_digest": (sha256(forecast["capacity_factor"].to_numpy().tobytes()).hexdigest()[:16]
                                    if forecast is not None else None),
                "fault_scenario": asdict(fault_scenario) if fault_scenario is not None else None,
                "fault_generator_version": _FAULT_GENERATOR_VERSION,
                "fault_schedule_digest": event_schedule_digest,
                "fault_events": raw_fault_events,
                "metadata": metadata or {}}
    case_id = sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
    equipment_sizing = build_equipment_sizing(
        energy, sizing, costed_short, costed_long, plant, strategy, perfect,
    )
    # The built plant needs its own register: reading the perfect-information one
    # while looking at an imperfect result would describe equipment that case never had.
    equipment_sizing_imperfect: tuple[dict[str, Any], ...] = ()
    if historic_reference is not None:
        equipment_sizing_imperfect = build_equipment_sizing(
            energy, historic_sizing, historic_costed_short, historic_costed_long,
            plant, strategy, historic_reference,
        )
    eb = None
    if baseline is not None and baseline_stores is not None:
        eb = calculate_economics(
            baseline, energy, historic_sizing.nominal_methane_kg_h, baseline_stores[0],
            baseline_stores[1], plant, economic, strategy, historic_electrolyser_kw,
        )
        # How much the scheduled plant gains over running flat out with no storage,
        # measured on the case the plant would actually realise. Both sides carry the
        # same weather information and the same faults, so the ratio isolates what
        # storage and scheduling are worth.
        realized = eif if eif is not None else ei if ei is not None else ep
        # Reported as baseline-over-realised so it reads the same way round as every
        # other reference-relative metric: a fraction at or below one, where a smaller
        # number means the reference falls further short of what this plant achieves.
        eb["baseline_methane_ratio"] = (
            eb["average_annual_methane_kg"] / realized["average_annual_methane_kg"]
            if realized["average_annual_methane_kg"] else None
        )
        eb["baseline_methane_ratio_basis"] = (
            "No storage, on-off, same faults / imperfect forecast + faults"
            if eif is not None else
            "No storage, on-off, no faults / imperfect forecast" if ei is not None else
            "No storage, on-off, no faults / perfect information"
        )
    result = CaseResult(
        case_id=case_id, energy=energy, sizing=sizing,
        perfect=perfect, imperfect=imperfect,
        economics_perfect=ep, economics_imperfect=ei,
        equipment_sizing=equipment_sizing,
        imperfect_with_faults=imperfect_with_faults,
        economics_imperfect_with_faults=eif,
        baseline=baseline, economics_baseline=eb,
        equipment_sizing_imperfect=equipment_sizing_imperfect,
    sizing_imperfect=historic_sizing if historic_reference is not None else None,
        imperfect_realised_deficit_kwh_per_year=realised_deficit,
    )
    if save_outputs:
        _report_progress(progress, "Outputs: writing case artifacts.")
        result.output_dir = save_case_outputs(result, payload, output_root)
    _report_progress(progress, f"Case {case_id}: calculations complete.")
    return result


def calibrate_capacity_factors(
        actual: pd.DataFrame, forecast: pd.DataFrame | None = None, *,
        plant: PlantParameters = PlantParameters(),
        thermal: ThermalParameters = ThermalParameters(),
        strategy: StrategyConfig = StrategyConfig(),
        short_storage_template: StorageParameters = StorageParameters(),
        long_storage_template: StorageParameters = StorageParameters(
            self_discharge_fraction_per_h=1e-5,
        ),
        economic: EconomicParameters = EconomicParameters(),
        factor_minimum: float = 0.0,
        factor_maximum: float = 2.0,
        factor_increment: float = 0.05,
        max_evaluations: int = MAX_CAPACITY_CALIBRATION_EVALUATIONS,
        progress: ProgressCallback | None = None,
) -> CapacityCalibrationResult:
    """Minimise perfect-information LCOM with a capped bounded pattern search."""
    if not math.isfinite(factor_minimum) or not math.isfinite(factor_maximum):
        raise ValueError("Calibration factor bounds must be finite")
    if factor_maximum <= factor_minimum:
        raise ValueError("Calibration factor maximum must exceed its minimum")
    if not math.isfinite(factor_increment) or factor_increment <= 0:
        raise ValueError("Calibration factor increment must be positive")
    if int(max_evaluations) != max_evaluations or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    max_evaluations = min(
        int(max_evaluations), MAX_CAPACITY_CALIBRATION_EVALUATIONS,
    )

    def snap(value: float) -> float:
        bounded = min(factor_maximum, max(factor_minimum, float(value)))
        units = round((bounded - factor_minimum) / factor_increment)
        return round(
            min(factor_maximum, factor_minimum + units * factor_increment), 10,
        )

    # Auto-calibration searches only the coupled process-overcapacity and
    # storage-energy design space. Storage transfer power is unconstrained.
    starting_factors = (
        snap(strategy.f_ocp),
        snap(strategy.f_socp_long),
    )
    cache: dict[tuple[float, float], dict[str, Any]] = {}

    def rank(record: dict[str, Any]) -> tuple[float, ...]:
        lcom = record["lcom"] if record["lcom"] is not None else math.inf
        if record["feasible"]:
            return 0.0, float(lcom)
        return 1.0, float(record["deficit"]), float(lcom)

    def evaluate(factors: tuple[float, float]) -> dict[str, Any] | None:
        key = (snap(factors[0]), snap(factors[1]))
        if key in cache:
            return cache[key]
        if len(cache) >= max_evaluations:
            return None
        number = len(cache) + 1
        _report_progress(
            progress,
            "Capacity calibration: perfect-information evaluation "
            f"{number}/{max_evaluations} at "
            f"f_OCP={key[0]:g}, f_SOCP={key[1]:g}.",
        )
        candidate_strategy = replace(
            strategy,
            f_ocp=key[0],
            f_socp_long=key[1],
        )
        candidate = run_case(
            actual, forecast, plant=plant, thermal=thermal, strategy=candidate_strategy,
            short_storage_template=short_storage_template,
            long_storage_template=long_storage_template, economic=economic,
            include_imperfect=False,
            use_forecast_commitment=forecast is not None,
        )
        lcom = candidate.economics_perfect.get("lcom_usd_per_kg_ch4")
        lcom_value = (
            float(lcom) if lcom is not None and math.isfinite(float(lcom)) else None
        )
        deficit = max(0.0, float(candidate.sizing.average_annual_balance_deficit_kwh))
        feasible = bool(candidate.sizing.feasible and deficit <= 1e-6 and lcom_value is not None)
        record = {
            "factors": key,
            "lcom": lcom_value,
            "deficit": deficit,
            "feasible": feasible,
        }
        cache[key] = record
        best_so_far = min(cache.values(), key=rank)
        candidate_lcom = (
            f"${lcom_value:,.4f}/kg" if lcom_value is not None else "undefined"
        )
        candidate_status = (
            "cyclically feasible" if feasible
            else f"infeasible; deficit={deficit / 1000:,.3f} MWh/year"
        )
        best_lcom = (
            f"${best_so_far['lcom']:,.4f}/kg"
            if best_so_far["lcom"] is not None else "undefined"
        )
        best_factors = best_so_far["factors"]
        _report_progress(
            progress,
            f"Capacity calibration: evaluation {number}/{max_evaluations} complete; "
            f"LCOM={candidate_lcom}, {candidate_status}. Best so far: "
            f"f_OCP={best_factors[0]:g}, f_SOCP={best_factors[1]:g}, "
            f"LCOM={best_lcom}.",
        )
        return record

    best = evaluate(starting_factors)
    if best is None:  # pragma: no cover - guarded by max_evaluations validation
        raise ModelError("Capacity calibration could not evaluate its starting point")
    starting_lcom = best["lcom"]
    span = factor_maximum - factor_minimum
    step_targets = (0.25, 0.10)
    steps = sorted({
        round(max(1, round(target / factor_increment)) * factor_increment, 10)
        for target in step_targets
        if target + 1e-12 >= factor_increment
    }, reverse=True)
    steps = [step for step in steps if step <= span + 1e-12]
    if not steps:
        steps = [factor_increment]

    completed_all_steps = True
    for step in steps:
        while True:
            base = best["factors"]
            improved = False
            for axis in range(2):
                for direction in (-1.0, 1.0):
                    values = list(base)
                    values[axis] = snap(values[axis] + direction * step)
                    candidate_record = evaluate(tuple(values))
                    if candidate_record is None:
                        completed_all_steps = False
                        break
                    if rank(candidate_record) < rank(best):
                        best = candidate_record
                        improved = True
                        break
                if improved or not completed_all_steps:
                    break
            if not improved:
                break
            if not completed_all_steps:
                break
        if not completed_all_steps:
            break

    optimized_factors = best["factors"]
    optimized_strategy = replace(
        strategy,
        f_ocp=optimized_factors[0],
        f_socp_long=optimized_factors[1],
    )
    limit_reached = len(cache) >= max_evaluations and not completed_all_steps
    _report_progress(
        progress,
        "Capacity calibration: selected "
        f"f_OCP={optimized_factors[0]:g}, f_SOCP={optimized_factors[1]:g} "
        f"after {len(cache)} evaluations"
        + (f"; {max_evaluations}-evaluation limit reached." if limit_reached else "."),
    )
    return CapacityCalibrationResult(
        strategy=optimized_strategy,
        starting_factors=starting_factors,
        optimized_factors=optimized_factors,
        starting_lcom_usd_per_kg_ch4=starting_lcom,
        optimized_lcom_usd_per_kg_ch4=best["lcom"],
        evaluations=len(cache),
        converged=completed_all_steps,
        limit_reached=limit_reached,
        feasible=bool(best["feasible"]),
        average_annual_balance_deficit_kwh=float(best["deficit"]),
    )


def save_case_outputs(result: CaseResult, inputs: dict[str, Any],
                      output_root: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write complete, reproducible artifacts for a model case."""
    case_dir = Path(output_root) / result.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    result.perfect.hourly.rename_axis("timestamp").to_csv(case_dir / "perfect_hourly.csv")
    result.perfect.daily.rename_axis("date").to_csv(case_dir / "perfect_daily.csv")
    if result.imperfect is not None:
        result.imperfect.hourly.rename_axis("timestamp").to_csv(case_dir / "imperfect_hourly.csv")
        result.imperfect.daily.rename_axis("date").to_csv(case_dir / "imperfect_daily.csv")
    if result.imperfect_with_faults is not None:
        result.imperfect_with_faults.hourly.rename_axis("timestamp").to_csv(
            case_dir / "imperfect_with_faults_hourly.csv"
        )
        result.imperfect_with_faults.daily.rename_axis("date").to_csv(
            case_dir / "imperfect_with_faults_daily.csv"
        )
    summary = {"case_id": result.case_id, "created_at": datetime.now(UTC).isoformat(),
               "energy": asdict(result.energy), "sizing": asdict(result.sizing),
               "perfect_metrics": result.perfect.metrics,
                "imperfect_metrics": result.imperfect.metrics if result.imperfect is not None else None,
                "imperfect_with_faults_metrics": (result.imperfect_with_faults.metrics
                                                   if result.imperfect_with_faults is not None else None),
                "economics_perfect": result.economics_perfect,
                "economics_imperfect": result.economics_imperfect,
                "economics_imperfect_with_faults": result.economics_imperfect_with_faults,
                "fault_events": (result.imperfect_with_faults.metadata.get("fault_events", [])
                                 if result.imperfect_with_faults is not None else []),
                "fault_effective_intervals": (
                    result.imperfect_with_faults.metadata.get("fault_effective_intervals", [])
                    if result.imperfect_with_faults is not None else []
                ),
               "capacity_calibration": result.perfect.metadata.get("capacity_calibration"),
               "equipment_sizing": list(result.equipment_sizing),
                "warnings": sorted(set(list(result.sizing.warnings) + result.perfect.warnings +
                                       (result.imperfect.warnings if result.imperfect is not None else []) +
                                       (result.imperfect_with_faults.warnings
                                        if result.imperfect_with_faults is not None else [])))}
    (case_dir / "inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    try:
        plotted = (result.imperfect_with_faults or result.imperfect or result.perfect)
        title = ("imperfect with faults" if result.imperfect_with_faults is not None
                 else "imperfect forecast" if result.imperfect is not None
                 else "perfect information")
        build_dispatch_figure(plotted, f"Case {result.case_id}: {title}").write_html(
            case_dir / "dispatch.html", include_plotlyjs="cdn")
        build_reactor_count_figure(
            plotted, f"Case {result.case_id}: {title} — reactor trains"
        ).write_html(case_dir / "reactor_counts.html", include_plotlyjs="cdn")
    except ImportError:  # pragma: no cover
        pass
    return case_dir


def build_dispatch_figure(
    simulation: SimulationResult,
    title: str = "Dispatch result",
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotly is required to build figures") from exc
    f = simulation.hourly
    if f.empty:
        raise ValueError("Cannot build a dispatch figure from an empty hourly result")
    # All model timestamps are uniformly spaced UTC hours. x0/dx preserves every
    # hourly point while avoiding eight duplicate arrays of 87,672 ISO timestamps
    # in the Dash response.
    time_axis = {"x0": f.index[0].isoformat(), "dx": 3_600_000}

    def trace(column: str, name: str, **kwargs):
        return go.Scattergl(
            **time_axis,
            y=f[column].to_numpy(dtype=float, copy=False),
            name=name,
            **kwargs,
        )

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    fig.add_trace(trace("actual_cf", "Actual CF"), row=1, col=1)
    fig.add_trace(trace("forecast_cf", "Forecast CF", line={"dash": "dot"}), row=1, col=1)
    fig.add_trace(trace("methane_kg", "Methane kg/h"), row=2, col=1, secondary_y=False)
    if "methane_delivery_kg" in f:
        fig.add_trace(
            trace(
                "methane_delivery_kg", "Continuous methane delivery",
                line={"dash": "dash", "color": "#7b3294"},
            ),
            row=2, col=1, secondary_y=False,
        )
    fig.add_trace(trace("curtailed_kwh", "Curtailment (kWh/h)"), row=2, col=1, secondary_y=False)
    fig.add_trace(trace("short_soc_kwh", "Short SOC"), row=3, col=1)
    fig.add_trace(trace("long_soc_kwh", "Long SOC"), row=3, col=1)
    if "planned_short_soc_kwh" in f:
        fig.add_trace(
            trace("planned_short_soc_kwh", "Planned short SOC", line={"dash": "dot"}),
            row=3, col=1,
        )
    if "planned_long_soc_kwh" in f:
        fig.add_trace(
            trace("planned_long_soc_kwh", "Planned long SOC", line={"dash": "dot"}),
            row=3, col=1,
        )
    if "long_h2_soc_kg" in f and float(f["long_h2_soc_kg"].max()) > 0:
        fig.add_trace(
            trace("long_h2_soc_kg", "Stored H2 (kg)", line={"color": "#7b3294"}),
            row=3, col=1, secondary_y=True,
        )
    if "long_co2_soc_kg" in f and float(f["long_co2_soc_kg"].max()) > 0:
        fig.add_trace(
            trace("long_co2_soc_kg", "Stored CO2 (kg)", line={"color": "#008837"}),
            row=3, col=1, secondary_y=True,
        )
    fault_colours = {
        "battery": "#4477AA", "hydrogen": "#66CCEE",
        "sabatier": "#EE6677", "dac": "#228833",
    }
    for interval in simulation.metadata.get("fault_effective_intervals", []):
        start = _utc_timestamp(interval["start"])
        end = _utc_timestamp(interval["end"])
        component = str(interval["component"])
        capacity = float(interval["capacity_fraction"])
        colour = fault_colours.get(component, "#777777")
        fig.add_vrect(
            x0=start, x1=end, fillcolor=colour, opacity=0.12,
            line_width=0, layer="below",
        )
        midpoint = start + (end - start) / 2
        fig.add_trace(go.Scatter(
            x=[midpoint], y=[1.0], mode="markers",
            marker={"size": 12, "color": colour, "opacity": 0.02},
            name=f"{component.title()} fault",
            showlegend=False,
            customdata=[[start.isoformat(), end.isoformat(), capacity,
                         float(interval["duration_h"]) ]],
            hovertemplate=(
                f"{component.title()} fault<br>Start: %{{customdata[0]}}"
                "<br>End: %{customdata[1]}<br>Retained capacity: %{customdata[2]:.0%}"
                "<br>Duration: %{customdata[3]:.0f} h<extra></extra>"
            ),
        ), row=1, col=1)
    fig.update_yaxes(title_text="Capacity factor", row=1, col=1)
    fig.update_yaxes(title_text="Production / curtailment", row=2, col=1)
    fig.update_yaxes(title_text="SOC (kWh)", row=3, col=1)
    fig.update_yaxes(title_text="Gas inventory (kg)", row=3, col=1, secondary_y=True)
    fig.update_xaxes(autorange=True, type="date")
    fig.update_layout(title=title, template="plotly_white", height=600,
                      hovermode="x unified")
    return fig


def seasonal_reactor_count_summary(simulation: SimulationResult) -> str | None:
    """Return a compact winter/spring/summer/autumn train-count summary."""
    strategy = simulation.metadata.get("strategy", {})
    if strategy.get("reactor_scheduling_mode") != "seasonal":
        return None
    if "planned_reactor_count" not in simulation.hourly or simulation.hourly.empty:
        return None
    season_months = {
        "Winter": (12, 1, 2),
        "Spring": (3, 4, 5),
        "Summer": (6, 7, 8),
        "Autumn": (9, 10, 11),
    }
    parts = []
    months = simulation.hourly.index.month
    planned = simulation.hourly["planned_reactor_count"]
    for label, included_months in season_months.items():
        values = planned[np.isin(months, included_months)]
        value = str(int(round(float(values.max())))) if not values.empty else "—"
        parts.append(f"{label} {value}")
    return " | ".join(parts)


def build_reactor_count_figure(
        simulation: SimulationResult,
        title: str = "Planned and active reactor trains",
):
    """Plot planned and realized hourly integer reactor-train counts separately."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plotly is required to build figures") from exc
    f = simulation.hourly
    if f.empty:
        raise ValueError("Cannot build a reactor-count figure from an empty hourly result")
    time_axis = {"x0": f.index[0].isoformat(), "dx": 3_600_000}

    def trace(column: str, name: str, **kwargs):
        return go.Scattergl(
            **time_axis,
            y=f[column].to_numpy(dtype=float, copy=False),
            name=name,
            **kwargs,
        )

    fig = go.Figure()
    if "planned_reactor_count" in f:
        seasonal_summary = seasonal_reactor_count_summary(simulation)
        planned_name = (
            f"Planned trains ({seasonal_summary})" if seasonal_summary else "Planned trains"
        )
        fig.add_trace(trace(
            "planned_reactor_count", planned_name, line={"dash": "dot"},
        ))
    if "actual_reactor_count" in f:
        fig.add_trace(trace(
            "actual_reactor_count", "Active trains", line_shape="hv",
        ))
    fig.update_yaxes(title_text="Reactor trains", dtick=1, rangemode="tozero")
    fig.update_xaxes(autorange=True, type="date", title_text="UTC time")
    fig.update_layout(title=title, template="plotly_white", height=390,
                      hovermode="x unified")
    return fig


LIMIT_CATEGORIES = ("At target", "Solar resource", "Plant capacity",
                    "Product storage full",
                    "DAC fault", "Hydrogen fault", "Sabatier fault", "Battery fault")
_FAULT_LIMIT_LABELS = {
    "dac_capacity_fraction": "DAC fault",
    "hydrogen_capacity_fraction": "Hydrogen fault",
    "sabatier_capacity_fraction": "Sabatier fault",
    "battery_capacity_fraction": "Battery fault",
}


def limiting_subsystem(hourly: pd.DataFrame, tolerance: float = 1e-9) -> pd.Series:
    """Label what held methane production back in each hour.

    A derated subsystem takes precedence, because an active fault is the binding
    constraint whatever else is happening. Failing that, an unmet methane target
    means the solar resource ran short, while curtailing surplus energy means the
    plant itself could not absorb what was available. Anything else met its target
    without wasting energy.
    """
    index = hourly.index
    labels = pd.Series("At target", index=index, dtype=object)

    curtailing = hourly.get("curtailed_kwh", pd.Series(0.0, index=index)) > tolerance
    shortfall = hourly.get("methane_shortfall_kg", pd.Series(0.0, index=index)) > tolerance
    labels = labels.mask(curtailing, "Plant capacity")
    labels = labels.mask(shortfall, "Solar resource")
    # A full product vessel curtails too, but for a reason the plant-capacity label
    # would hide: the process kept up and the tank did not. It outranks both, because
    # an hour that spilled methane was constrained by the vessel whatever else was
    # also true of it.
    vessel_full = hourly.get("methane_curtailed_kg", pd.Series(0.0, index=index)) > tolerance
    labels = labels.mask(vessel_full, "Product storage full")

    # The most severely derated subsystem wins when several are faulted at once.
    worst = pd.Series(1.0, index=index, dtype=float)
    for column, label in _FAULT_LIMIT_LABELS.items():
        if column not in hourly:
            continue
        fraction = hourly[column].astype(float)
        binding = (fraction < 1.0 - tolerance) & (fraction < worst)
        labels = labels.mask(binding, label)
        worst = worst.where(~binding, fraction)
    return labels


def daily_limiting_subsystem(hourly: pd.DataFrame) -> pd.Series:
    """Reduce the hourly labels to one dominant limit per day.

    "At target" only wins a day when nothing else constrained it, so a day with any
    real limitation is reported by that limitation rather than by its quiet hours.
    """
    labels = limiting_subsystem(hourly)
    frame = pd.DataFrame({"day": labels.index.normalize(), "label": labels.to_numpy()})
    counts = frame.groupby(["day", "label"]).size().rename("hours").reset_index()
    constrained = counts[counts["label"] != "At target"]
    ranked = constrained if not constrained.empty else counts
    best = ranked.sort_values(["day", "hours"], ascending=[True, False])
    daily = best.groupby("day").first()["label"]
    # Days with no constrained hours at all fall back to "At target".
    everything = pd.Index(sorted(frame["day"].unique()), name="day")
    return daily.reindex(everything, fill_value="At target")


def make_synthetic_weather(start: str = "2010-01-01", years: int = 1, *,
                           latitude_deg: float = 50.0, cloud_variability: float | None = None,
                           seed: int = 7,
                           parameters: SyntheticWeatherParameters = SyntheticWeatherParameters()) -> pd.DataFrame:
    """Deterministic demo data for tests and offline notebook/app use.

    The clear-sky profile comes from solar geometry, so latitude sets both the
    length of the day and the height of the sun: declination follows the axial
    tilt through the year, irradiance on the array is the cosine of the angle of
    incidence on a fixed equator-facing plane, and the beam is attenuated by air
    mass.  Polar night and the reversed southern-hemisphere seasons therefore
    fall out of the geometry rather than being imposed.

    Cloud cover is a single latitude-independent ``clear_sky_index`` plus daily
    noise, so this profile carries no regional cloud climatology: it will
    under-state how much sunnier southern Europe is than the maritime
    north-west.  Use the Renewables.ninja path for siting claims that depend on
    that difference.
    """
    start_stamp = pd.Timestamp(start, tz="UTC")
    index = pd.date_range(start_stamp, start_stamp + pd.DateOffset(years=years), freq="h", inclusive="left")
    rng = np.random.default_rng(seed)
    doy, hour = index.dayofyear.to_numpy(), index.hour.to_numpy()
    cloud_variability = (parameters.cloud_variability
                         if cloud_variability is None else cloud_variability)
    declination = np.radians(parameters.axial_tilt_deg) * np.sin(
        2 * np.pi * (doy - parameters.solar_seasonal_phase_day)
        / parameters.seasonal_period_days
    )
    latitude = np.radians(latitude_deg)
    # A fixed array is tilted towards the equator, so the plane it presents to
    # the sun behaves like a horizontal surface at a lower effective latitude.
    array_latitude = latitude - np.sign(latitude) * np.radians(parameters.array_tilt_deg)
    hour_angle = np.pi * (hour - parameters.solar_noon_hour) / 12.0
    sun_elevation_sine = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    incidence_cosine = (
        np.sin(array_latitude) * np.sin(declination)
        + np.cos(array_latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    sun_up = sun_elevation_sine > parameters.minimum_sun_elevation_sine
    air_mass = np.where(sun_up, 1.0 / np.where(sun_up, sun_elevation_sine, 1.0), 0.0)
    beam_fraction = np.where(
        sun_up, parameters.atmospheric_transmittance ** (air_mass ** AIR_MASS_EXPONENT), 0.0
    )
    clear = np.where(sun_up, np.clip(incidence_cosine, 0.0, None), 0.0) * beam_fraction
    seasonal = parameters.clear_sky_index
    daily_cloud = rng.normal(1.0, cloud_variability, len(pd.Index(index.floor("D")).unique()))
    day_codes = pd.factorize(index.floor("D"))[0]
    cf = np.clip(
        clear * seasonal * daily_cloud[day_codes], 0.0,
        parameters.maximum_capacity_factor,
    )
    ambient = (
        parameters.ambient_mean_temperature_k
        + parameters.ambient_seasonal_amplitude_k
        * np.sin(
            2 * np.pi * (doy - parameters.ambient_seasonal_phase_day)
            / parameters.seasonal_period_days
        )
        + parameters.ambient_diurnal_amplitude_k
        * np.sin(
            2 * np.pi * (hour - parameters.ambient_diurnal_phase_hour)
            / parameters.diurnal_period_hours
        )
    )
    return pd.DataFrame({"capacity_factor": cf, "ambient_temperature_k": ambient}, index=index)


__all__ = ["CapacityCalibrationResult", "CaseResult", "EconomicParameters", "FaultDistribution",
           "FaultEvent", "FaultScenario", "ModelError", "PlantEnergyResult",
           "PlantParameters", "SimulationResult", "StorageParameters", "StorageSizingResult",
           "StrategyConfig", "SyntheticWeatherParameters", "ThermalParameters", "WeatherConfig", "WeatherDataError",
           "build_climatology_forecast", "build_dispatch_figure", "build_reactor_count_figure",
           "build_target_schedule",
           "build_equipment_sizing",
           "LIMIT_CATEGORIES", "cached_weather_sites", "cached_weather_years",
    "calculate_economics", "daily_limiting_subsystem", "limiting_subsystem", "calculate_plant_energy",
           "calibrate_capacity_factors",
           "fetch_solar_profile", "generate_battery_faults", "generate_dac_faults",
           "generate_fault_events", "generate_hydrogen_faults", "generate_sabatier_faults",
           "load_costs", "seasonal_reactor_count_summary",
           "size_methane_storage",
           "discover_latest_weather_year",
           "make_synthetic_weather", "nominal_methane_rate", "round_reactor_equivalents", "run_case",
           "sabatier_train_thermal_properties",
           "save_case_outputs", "simulate_dispatch", "size_perfect_storage", "split_weather_period",
           "validate_hourly_profile"]
