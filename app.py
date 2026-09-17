"""Lightweight Dash interface for the solar-to-methane model."""

from __future__ import annotations

import gc
import base64
from datetime import datetime
import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock, Thread
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from pfd import format_number, render_pfd, summarize_streams
from ninja_usage import usage_snapshot
from result_files import (JOB_FIELDS, MAX_FILE_BYTES, collect_weather_cache,
                          describe_export_size, export_results, import_results,
                          normalize_showcase_ratio, restore_weather_cache, safe_inputs)

from model import (
    MAX_CAPACITY_CALIBRATION_EVALUATIONS,
    EconomicParameters, FaultDistribution, FaultEvent, FaultScenario,
    PlantParameters, StorageParameters, StrategyConfig,
    SyntheticWeatherParameters, ThermalParameters, WeatherConfig,
    WeatherDataError, _load_api_token,
    build_climatology_forecast, build_dispatch_figure, build_reactor_count_figure,
    calibrate_capacity_factors, fetch_solar_profile,
    LIMIT_CATEGORIES, cached_weather_sites, daily_limiting_subsystem,
    generate_fault_events, make_synthetic_weather, run_case,
    seasonal_reactor_count_summary, split_weather_period,
)

SHOWCASE_PATH = Path("data") / "showcase" / "showcase_results.csv"
# Pre-computed results shipped with the repository so the dashboard can be explored
# without running a simulation or holding a weather API key. Rebuild with
# scripts/build_demo.py whenever the model changes enough to make them misleading.
DEMO_DIR = Path("data") / "demo"
# Two showcases over the same ten sites, differing only in how many reactor trains
# run in parallel. They share site names, so loading one replaces the other on the
# map rather than doubling the dots.
DEMO_SHOWCASES = {
    "parallel": (DEMO_DIR / "showcase-parallel.json.gz", "four parallel trains"),
    "single": (DEMO_DIR / "showcase-single.json.gz", "a single train"),
}
DEMO_SINGLE_SITE = DEMO_DIR / "single_site.json.gz"
DEMO_FILES = (DEMO_SHOWCASES["parallel"][0], DEMO_SINGLE_SITE)
# Default folder for server-side result saves; outputs/ is already gitignored.
DEFAULT_SAVE_DIR = Path("outputs")
RUN_JOBS: dict[str, dict] = {}
RUN_JOBS_LOCK = Lock()
SHOWCASE_RESULTS: dict[str, dict] = {}
OPERATING_STRATEGIES = ("through_night", "limping", "hard_shutdown")
# Four ways to carry energy from summer to winter. The first three store it upstream of
# the reactor and let the plant keep making methane through the dark months. The fourth
# stores the methane itself: the plant follows the sun and a product vessel holds the
# summer surplus to keep the gate flow constant. That store is one-way — it cannot power
# the plant — which is what makes it a different strategy rather than a cheaper tank.
LONG_STORAGE_METHODS = ("battery", "hydrogen", "h2_co2", "methane")
LONG_STORAGE_LABELS = {
    "battery": "Battery",
    "hydrogen": "Hydrogen",
    "h2_co2": "H2 + CO2 gas storage",
    "methane": "Methane product storage",
}
# The method whose seasonal store is the product vessel rather than an energy store.
PRODUCT_STORAGE_METHOD = "methane"
MAX_COMPLETED_JOBS = 2
DASHBOARD_HOURLY_COLUMNS = (
    "actual_cf", "forecast_cf", "methane_kg", "curtailed_kwh",
    # limiting_subsystem reads the shortfall to tell "the sun ran short" apart from
    # "the plant could not absorb it". Without it the fallback is silently zero and
    # "Solar resource" can never be reported.
    "methane_shortfall_kg",
    "methane_delivery_kg", "methane_storage_inventory_kg",
    "methane_storage_inflow_kg", "methane_storage_outflow_kg",
    "methane_storage_compressor_load_kwh", "methane_storage_expander_generation_kwh",
    "methane_storage_expander_reheat_kwh",
    "continuous_reactor_equivalents", "planned_reactor_count", "actual_reactor_count",
    "short_soc_kwh", "planned_short_soc_kwh", "long_soc_kwh", "planned_long_soc_kwh",
    "long_h2_soc_kg", "long_co2_soc_kg", "long_h2_charge_kg", "long_co2_charge_kg",
    "long_fuel_cell_h2_kg", "long_direct_h2_kg", "long_direct_co2_kg",
    "long_h2_storage_compressor_load_kwh",
    "long_h2_storage_expander_generation_kwh",
    "long_h2_storage_expander_reheat_kwh",
    "long_co2_storage_compressor_load_kwh",
    "long_co2_storage_expander_generation_kwh",
    "long_co2_storage_expander_reheat_kwh",
    "carbonator_temperature_k",
    "calciner_temperature_k", "sabatier_mean_temperature_k",
    "battery_capacity_fraction", "hydrogen_capacity_fraction",
    "sabatier_capacity_fraction", "dac_capacity_fraction",
    "short_inaccessible_kwh", "long_inaccessible_kwh",
)
# The subset any plot, table or summary reads back out of a loaded bundle. Everything
# else in DASHBOARD_HOURLY_COLUMNS is consumed during the run — by economics sizing and
# by summarize_streams — and its conclusions are already frozen into the economics
# dataclasses and metadata["pfd_summary"], so nothing recomputes from those columns
# afterwards. A demo bundle exists only to be looked at, so it ships this subset;
# a run you saved yourself keeps every column.
DISPLAY_HOURLY_COLUMNS = (
    "actual_cf", "forecast_cf", "methane_kg", "methane_delivery_kg",
    "curtailed_kwh", "methane_shortfall_kg",
    "short_soc_kwh", "planned_short_soc_kwh",
    "long_soc_kwh", "planned_long_soc_kwh",
    "long_h2_soc_kg", "long_co2_soc_kg",
    "planned_reactor_count", "actual_reactor_count",
    "battery_capacity_fraction", "hydrogen_capacity_fraction",
    "sabatier_capacity_fraction", "dac_capacity_fraction",
)
# Showcase map metrics. Each entry carries the radio label and the one-line
# description shown on hover. Simple metrics are always offered; advanced metrics
# only appear once "Show advanced metrics" is ticked.
SIMPLE_METRIC_META = {
    "ch4_perfect": (
        "Annual CH4 (perfect info, no faults)",
        "Methane produced when the plant knows the weather in advance and nothing "
        "breaks. The theoretical ceiling for the site.",
    ),
    "ch4_imperfect": (
        "Annual CH4 (imperfect info, no faults)",
        "Methane produced when the plant commits against a climatology forecast "
        "instead of the actual weather. The gap from perfect is the forecast penalty.",
    ),
    "ch4_faults": (
        "Annual CH4 (imperfect info, with faults)",
        "Methane produced with both forecast error and simulated equipment outages. "
        "The most realistic production figure.",
    ),
    "lcom_perfect": (
        "Est. LCOM (perfect info, no faults)",
        "Illustrative levelised cost of methane under perfect foresight and no "
        "faults. Lower is better.",
    ),
    "lcom_imperfect": (
        "Est. LCOM (imperfect info, no faults)",
        "Illustrative levelised cost once the plant must rely on a forecast, with "
        "no equipment outages.",
    ),
    "lcom_faults": (
        "Est. LCOM (imperfect info, with faults)",
        "Illustrative levelised cost including both forecast error and simulated "
        "equipment outages.",
    ),
    "production_ratio": (
        "Forecast + fault production ratio",
        "Realised methane divided by the perfect-information ideal. 1.0 means "
        "forecast error and faults cost no production. It can exceed 1.0: the "
        "realised case runs a plant sized on the training record, which can carry "
        "more throughput than the perfect-information plant and so out-produce it "
        "in kilograms while costing more capital to do so. The cost ratio is where "
        "that shows up.",
    ),
    "forecast_cost_ratio": (
        "Forecast + fault cost ratio",
        "Realised LCOM divided by the perfect-information LCOM. Read it with care: "
        "the two cases are not the same plant. The realised case is sized on the "
        "five-year training record, the perfect-information case on the whole "
        "evaluation period, so this ratio carries the capital difference between "
        "those two plants as well as the cost of forecast error and faults — and "
        "across the shipped sites the capital difference dominates. A training "
        "window that missed a hard winter builds a cheaper plant and can push this "
        "below 1.0. Use the production ratio to judge forecast quality.",
    ),
    "baseline_production_ratio": (
        "No-storage baseline production ratio",
        "The same plant run flat out with no storage and simple on-off control, "
        "divided by what this plant realises, faults included on both sides. 0.5 "
        "means the do-nothing baseline manages half the output, so storage and "
        "scheduling double it.",
    ),
}
ADVANCED_METRIC_META = {
    "short_storage_capacity_mwh": (
        "Short-term storage capacity (MWh)",
        "Installed within-day store, sized from the daily state-of-charge range.",
    ),
    "long_storage_capacity_mwh": (
        "Long-term storage capacity (MWh)",
        "Installed seasonal store after f_SOCP is applied to the required capacity.",
    ),
    "methane_storage_capacity_kg": (
        "Storage capacity (CH4 product, kg)",
        "Product methane buffer needed to deliver a constant output from variable "
        "production.",
    ),
    "plant_utilisation": (
        "Plant utilisation",
        "Fraction of available operating hours in which the process actually ran.",
    ),
    "curtailment_fraction": (
        "Curtailment fraction",
        "Share of generated solar energy that could not be used or stored.",
    ),
    "total_capex_usd": (
        "Total CAPEX (dummy USD)",
        "Illustrative installed capital cost of the whole plant. Placeholder costs.",
    ),
    "annual_opex_usd_per_year": (
        "Annual OPEX (dummy USD/y)",
        "Illustrative fixed plus variable operating cost per year. Placeholder costs.",
    ),
    "annual_balance_deficit_mwh": (
        "Cyclic energy deficit (MWh/y)",
        "Energy by which the yearly cycle fails to close. Above zero means the "
        "selected capacity factors cannot sustain the plan.",
    ),
}
METRIC_META = {**SIMPLE_METRIC_META, **ADVANCED_METRIC_META}
# Carried through to the map frame but not offered as colour metrics: the combined
# figures the bundled CSV and saved files still use.
LEGACY_SHOWCASE_COLUMNS = (
    "average_annual_methane_kg", "lcom_usd_per_kg_ch4", "storage_capacity_mwh",
)
METRIC_LABELS = {key: label for key, (label, _) in METRIC_META.items()}
METRIC_DESCRIPTIONS = {key: text for key, (_, text) in METRIC_META.items()}
# The production ratio is the metric that isolates what imperfect information
# actually costs: it holds between 0.92 and 0.96 across all ten sites, while the
# cost ratio swings from 0.75 to 1.14 on sizing-window luck. See the cost ratio
# description for why.
DEFAULT_MAP_METRIC = "production_ratio"
# Dot area always tracks realised production, independent of the colour metric, so
# the map keeps a stable sense of scale as you switch metrics.
MAP_SIZE_COLUMN = "average_annual_methane_kg"
# Result dots have to stay a larger hover target than a coordinate-grid point, or the
# grid catches the pointer first. The grid is drawn at MAP_GRID_SIZE.
MAP_RESULT_SIZE_MAX = 30
MAP_RESULT_SIZE_MIN = 14
# Shown on the hover for any site whose plant, sized on the training record, cannot
# close its cycle on the decade it was then evaluated over.
DEFICIT_NOTE = ("Imperfect weather forecasting results in a net energy deficit "
                "at this f_OCP/f_SOCP level")
MAP_GRID_SIZE = 11
CASE_COMPARISON_METRICS = {
    "lcom_usd_per_kg_ch4": "LCOM (illustrative USD/kg CH4)",
    "average_annual_methane_kg": "Annual methane production (kg/year)",
    "total_capex_usd": "Total CAPEX (illustrative USD)",
    "annual_opex_usd_per_year": "Annual OPEX (illustrative USD/year)",
    "plant_utilisation_percent": "Plant utilisation (%)",
    "curtailment_percent": "Solar curtailment (%)",
    "average_annual_methane_shortfall_kg": "Annual methane shortfall (kg/year)",
    "annual_balance_deficit_mwh": "Cyclic energy deficit (MWh/year)",
    "forced_shutdown_hours": "Forced shutdown hours",
    "short_storage_capacity_mwh": "Installed short-storage energy (MWh)",
    "long_storage_capacity_mwh": "Installed long-storage energy (MWh carrier energy)",
    "short_storage_power_mw": "Peak short-storage transfer power (MW)",
    "long_storage_power_mw": "Peak long-storage transfer power (MW)",
    "methane_storage_capacity_kg": "Product methane buffer capacity (kg CH4)",
    "methane_storage_capex_usd": "Product methane buffer CAPEX (illustrative USD)",
    "forecast_cost_ratio": "LCOM imperfect / LCOM perfect",
}
CASE_COLOURS = {
    "through_night|battery": "#4477AA",
    "through_night|hydrogen": "#EE6677",
    "limping|battery": "#228833",
    "limping|hydrogen": "#CCBB44",
    "hard_shutdown|battery": "#66CCEE",
    "hard_shutdown|hydrogen": "#AA3377",
    "through_night|h2_co2": "#44AA99",
    "limping|h2_co2": "#882255",
    "hard_shutdown|h2_co2": "#999933",
    "through_night|methane": "#DDCC77",
    "limping|methane": "#332288",
    "hard_shutdown|methane": "#CC6677",
}
ALL_CASE_KEYS = tuple(
    f"{short_name}|{long_method}"
    for short_name in OPERATING_STRATEGIES
    for long_method in LONG_STORAGE_METHODS
)

PLANT_ASSUMPTION_META = {
    "solar_farm_mw": ("MW", "Solar farm nameplate capacity; this default is replaced by the dashboard input."),
    "electrolyser_kwh_per_kg_h2": ("kWh/kg H2", "Electricity consumed by the electrolyser per kilogram of hydrogen produced."),
    "fuel_cell_kwh_per_kg_h2": ("kWh/kg H2", "Bus electricity delivered by the fuel cell per kilogram of hydrogen consumed."),
    "fan_kwh_per_kg_co2": ("kWh/kg CO2", "DAC fan electricity required per kilogram of carbon dioxide captured."),
    "sabatier_heat_kwh_per_kg_ch4": ("kWh/kg CH4", "Sabatier reaction heat routed directly to feed heating before electric heat is used."),
    "co2_outlet_pressure_bar": ("bar", "Carbon-dioxide pressure required at the compressor outlet."),
    "compressor_isentropic_efficiency": ("fraction", "Isentropic efficiency applied to carbon-dioxide compression."),
    "compressor_stages": ("stages", "Number of intercooled carbon-dioxide compressor stages."),
    "intercool_temperature_k": ("K", "Carbon-dioxide temperature restored between compressor stages."),
    "sabatier_temperature_k": ("K", "Nominal Sabatier reactor operating temperature."),
    "carbonator_temperature_k": ("K", "Nominal calcium-looping carbonator operating temperature. Carbonation only proceeds while the CaO/CaCO3 equilibrium pressure stays below the CO2 partial pressure in the feed air, which caps this at roughly 790 K for 400 ppm air; a run above that limit is reported as a warning."),
    "calciner_temperature_k": ("K", "Nominal calcium-looping calciner operating temperature."),
    "reference_ambient_temperature_k": ("K", "Reference ambient temperature used in process heat balances."),
    "air_exhaust_hx_effectiveness": ("fraction", "Fraction of the carbonator air-preheat duty recovered from the depleted exhaust. The DAC air stream has by far the largest heat-capacity flow in the plant, so E_req scales with the unrecovered remainder and this is the most influential single assumption in the model."),
    "solids_heat_recovery_efficiency": ("fraction", "Fraction of sensible heat recovered from circulating solids."),
    "air_co2_mole_fraction": ("mol/mol", "Carbon-dioxide mole fraction assumed in ambient air."),
    "dac_capture_efficiency": ("fraction", "Single-pass fraction of incoming carbon dioxide removed by the DAC contactor; processed air mass, and therefore the carbonator air-heating duty, scales inversely with this value."),
    "air_pressure_bar": ("bar", "Pressure used for live CoolProp air-enthalpy calculations."),
    "air_fallback_cp_kwh_per_kg_k": ("kWh/(kg K)", "Fallback air heat capacity used only if CoolProp cannot evaluate the state."),
    "air_molar_mass_kg_per_mol": ("kg/mol", "Mean molar mass used for ambient air."),
    "cao_cp_kwh_per_kg_k": ("kWh/(kg K)", "Specific heat capacity used for calcium oxide."),
    "caco3_cp_kwh_per_kg_k": ("kWh/(kg K)", "Specific heat capacity used for calcium carbonate."),
    "solids_cycle_time_h": ("h", "Residence-time proxy used to infer solids inventory pending an explicit kinetic model."),
    "calcination_delta_h_j_per_mol": ("J/mol", "Molar enthalpy change used for calcium-carbonate calcination."),
    "calcination_delta_s_j_per_mol_k": ("J/(mol K)", "Molar entropy change combined with enthalpy and calciner temperature to calculate equilibrium pressure at the CO2 compressor inlet."),
    "ramp_fraction_per_h": ("nominal/h", "Maximum hourly change in the methane production setpoint relative to nominal output."),
    "hydrogen_storage_pressure_bar": ("bar", "Nominal hydrogen storage-vessel pressure; hydrogen is compressed from the process line to this pressure."),
    "co2_storage_pressure_bar": ("bar", "Nominal carbon-dioxide storage-vessel pressure; CO2 is compressed from the process line to this pressure."),
    "methane_storage_pressure_bar": ("bar", "Nominal product-methane storage-vessel pressure."),
    "methane_delivery_pressure_bar": ("bar", "Methane delivery pressure downstream of the storage expander."),
    "storage_expander_isentropic_efficiency": ("fraction", "Isentropic efficiency shared by the hydrogen, CO2, and methane storage expanders."),
    "storage_machine_stages": ("stages", "Number of equal pressure-ratio stages used for gas-storage compression and expansion."),
}

PHYSICAL_PROPERTY_ASSUMPTION_META = {
    "methane_molar_mass_kg_per_mol": ("kg/mol", "Molar mass of methane used as the stoichiometric production basis."),
    "hydrogen_molar_mass_kg_per_mol": ("kg/mol", "Molar mass of hydrogen used in Sabatier stoichiometry."),
    "carbon_dioxide_molar_mass_kg_per_mol": ("kg/mol", "Molar mass of carbon dioxide used in capture and methanation balances."),
    "water_molar_mass_kg_per_mol": ("kg/mol", "Molar mass of product water used in the reported mass balance."),
    "calcium_oxide_molar_mass_kg_per_mol": ("kg/mol", "Molar mass of calcium oxide used for circulating-solids inventory."),
    "calcium_carbonate_molar_mass_kg_per_mol": ("kg/mol", "Molar mass of calcium carbonate used for circulating-solids inventory."),
    "gas_constant_j_per_mol_k": ("J/(mol K)", "Universal gas constant used in the calcination equilibrium estimate."),
    "hydrogen_lhv_kwh_per_kg": ("kWh/kg H2", "Hydrogen lower heating value used for storage energy and tank sizing."),
    "co2_compression_fallback_cp_j_per_kg_k": ("J/(kg K)", "CO2 heat capacity used by the ideal-gas compressor fallback."),
    "co2_compression_fallback_gamma": ("dimensionless", "CO2 heat-capacity ratio used by the ideal-gas compressor fallback."),
    "co2_sensible_fallback_cp_j_per_kg_k": ("J/(kg K)", "CO2 heat capacity used when CoolProp sensible enthalpy is unavailable."),
    "h2_sensible_fallback_cp_j_per_kg_k": ("J/(kg K)", "Hydrogen heat capacity used when CoolProp sensible enthalpy is unavailable."),
    "hydrogen_compression_fallback_gamma": ("dimensionless", "Hydrogen heat-capacity ratio used by the ideal-gas storage-machinery fallback."),
    "methane_compression_fallback_cp_j_per_kg_k": ("J/(kg K)", "Methane heat capacity used by the ideal-gas storage-machinery fallback."),
    "methane_compression_fallback_gamma": ("dimensionless", "Methane heat-capacity ratio used by the ideal-gas storage-machinery fallback."),
}

THERMAL_ASSUMPTION_META = {
    "calciner_ua_kw_per_k": ("kW/K", "Overall heat-loss conductance of the calciner thermal node."),
    "carbonator_ua_kw_per_k": ("kW/K", "Overall heat-loss conductance of the carbonator thermal node."),
    "fixed_hot_auxiliary_kw": ("kW", "Constant electrical heat input added while the calciner and carbonator thermal nodes are kept hot."),
    "ambient_fallback_k": ("K", "Ambient temperature used only when the selected weather series has no temperature value."),
    "sabatier_reference_capacity_kg_ch4_h": ("kg CH4/h", "Reference Sabatier train throughput used to scale thermal properties."),
    "sabatier_reference_thermal_capacity_kwh_per_k": ("kWh/K", "Sabatier train thermal capacity at the reference throughput."),
    "sabatier_reference_ua_kw_per_k": ("kW/K", "Sabatier train heat-loss conductance at the reference throughput."),
    "sabatier_ua_scaling_exponent": ("dimensionless", "Exponent used to scale Sabatier heat-loss conductance with train throughput."),
}

STRATEGY_ASSUMPTION_META = {
    "short_strategy": ("—", "Default short-horizon thermal operating strategy; selectable in the dashboard."),
    "long_strategy": ("—", "Long-horizon policy used to target constant methane output."),
    "f_ocp": ("fraction", "Reserve denominator that reduces rated methane capacity relative to the fixed solar farm."),
    "f_socp_long": ("installed/required", "Installed long-storage energy divided by its calculated requirement."),
    "daylight_cf_cutoff": ("capacity factor", "Solar capacity-factor threshold used to distinguish daylight hours."),
    "parallel_reactor_count": ("trains", "Number of equal parallel Sabatier trains selected on the Single site tab."),
    "reactor_scheduling_mode": ("mode", "Selects storage-aware daily switching or fixed seasonal reactor-count targets."),
    "storage_planning_lookahead_days": ("days", "Forward forecast window used to preserve storage before committing the upper whole-train count."),
    "product_storage": ("on/off", "Whether methane is buffered to a constant gate flow. Off, output floats with the weather and the buffer vessel, its compressor and its expander are not built."),
}

WEATHER_ASSUMPTION_META = {
    "dataset": ("—", "Renewables.ninja reanalysis dataset used for API weather runs."),
    "system_loss": ("fraction", "Fractional photovoltaic system loss applied by the weather-data request."),
    "tracking": ("mode", "PV tracking mode supplied to Renewables.ninja; zero means fixed tilt."),
    "tilt": ("degrees", "Fixed photovoltaic array tilt from horizontal."),
    "azim": ("degrees", "Photovoltaic array azimuth, with 180 degrees facing south."),
    "training_years": ("years", "Historical period used to build the hourly climatology forecast."),
    "evaluation_years": ("years", "Out-of-sample period requested for API evaluation. "
                         "Every hour of it is kept for the dispatch plots, so a long "
                         "period makes a large saved file: roughly 10 MB per evaluation "
                         "year per result category, and comparison runs carry three or "
                         "four categories. Past about twenty years a save will not fit."),
}

SYNTHETIC_WEATHER_ASSUMPTION_META = {
    "cloud_variability": ("standard deviation", "Daily multiplicative cloud-factor variability in the offline profile."),
    "clear_sky_index": ("fraction", "Mean fraction of clear-sky irradiance realised after cloud cover."),
    "atmospheric_transmittance": ("fraction", "Clear-sky beam transmittance at zenith, before air-mass attenuation."),
    "axial_tilt_deg": ("degrees", "Earth's axial tilt, setting the annual swing of solar declination."),
    "array_tilt_deg": ("degrees", "Fixed photovoltaic array tilt towards the equator in the offline profile."),
    "solar_seasonal_phase_day": ("day of year", "Day on which solar declination crosses zero going north."),
    "seasonal_period_days": ("days", "Period used for annual solar and temperature sinusoids."),
    "solar_noon_hour": ("UTC hour", "Hour of local solar noon in the offline profile."),
    "minimum_sun_elevation_sine": ("sine of angle", "Sun elevation below which the offline profile reports no output."),
    "maximum_capacity_factor": ("capacity factor", "Upper clipping limit for offline photovoltaic capacity factor."),
    "ambient_mean_temperature_k": ("K", "Annual mean ambient temperature in the offline profile."),
    "ambient_seasonal_amplitude_k": ("K", "Amplitude of the offline annual ambient-temperature cycle."),
    "ambient_seasonal_phase_day": ("day of year", "Phase offset of the annual ambient-temperature sinusoid."),
    "ambient_diurnal_amplitude_k": ("K", "Amplitude of the offline daily ambient-temperature cycle."),
    "ambient_diurnal_phase_hour": ("UTC hour", "Phase offset of the daily ambient-temperature sinusoid."),
    "diurnal_period_hours": ("h", "Period used for the synthetic daily temperature sinusoid."),
}

STORAGE_ASSUMPTION_META = {
    "self_discharge_fraction_per_h": ("fraction/h", "Fraction of stored energy lost during each hour."),
    "co2_self_discharge_fraction_per_h": ("fraction/h", "Fraction of the stored carbon-dioxide inventory lost during each hour."),
    "initial_soc_fraction": ("fraction", "Opening state of charge; replaced by the dashboard input for every store."),
}

ECONOMIC_ASSUMPTION_META = {
    "metadata.currency": ("—", "Currency in which all monetary assumptions and results are expressed."),
    "metadata.base_year": ("year", "Price basis year for the illustrative cost assumptions."),
    "financial.project_life_years": ("years", "Operating life over which annualised project cost is calculated."),
    "financial.real_discount_rate": ("fraction", "Real discount rate used in the capital-recovery factor."),
    "solar.capex_usd_per_kw": ("USD/kW", "Installed solar photovoltaic capital cost per unit of nameplate power."),
    "solar.fixed_opex_fraction_per_year": ("CAPEX/year", "Annual fixed solar operating cost as a fraction of solar CAPEX."),
    "battery.energy_capex_usd_per_kwh": ("USD/kWh", "Battery energy-capacity capital cost."),
    "battery.power_capex_usd_per_kw": ("USD/kW", "Battery charge/discharge power capital cost."),
    "battery.fixed_opex_fraction_per_year": ("CAPEX/year", "Annual fixed battery operating cost as a fraction of battery CAPEX."),
    "hydrogen_storage.tank_capex_usd_per_kg": ("USD/kg H2", "Hydrogen tank capital cost per kilogram of storage capacity."),
    "hydrogen_storage.extra_electrolyser_capex_usd_per_kw": ("USD/kW", "Capital cost of additional electrolyser power used for hydrogen storage."),
    "hydrogen_storage.fixed_opex_fraction_per_year": ("CAPEX/year", "Annual fixed hydrogen-storage operating cost as a fraction of vessel, charging, compression, expansion, and fuel-cell CAPEX."),
    "co2_storage.tank_capex_usd_per_kg": ("USD/kg CO2", "Illustrative carbon-dioxide pressure-vessel capital cost per kilogram of capacity."),
    "co2_storage.fixed_opex_fraction_per_year": ("CAPEX/year", "Annual fixed CO2-storage operating cost as a fraction of CO2 vessel, compressor, and expander CAPEX."),
    "methane_storage.tank_capex_usd_per_kg": ("USD/kg CH4", "Illustrative methane buffer-vessel capital cost per kilogram of capacity."),
    "methane_storage.initial_fill_usd_per_kg": ("USD/kg CH4", "Illustrative cost of the methane the vessel is commissioned holding. The plant delivers against that stock before it has made anything, so it is bought at commissioning rather than treated as free product."),
    "methane_storage.fixed_opex_fraction_per_year": ("CAPEX/year", "Annual fixed methane-buffer operating cost as a fraction of vessel, compressor, and expander CAPEX."),
    "plant_opex.fixed_fraction_of_plant_capex_per_year": ("CAPEX/year", "Annual fixed plant operating cost as a fraction of process-plant CAPEX."),
    "plant_opex.variable_usd_per_kg_ch4": ("USD/kg CH4", "Variable process operating cost per kilogram of methane produced."),
}

PLANT_UNIT_DESCRIPTIONS = {
    "electrolyser": "water electrolysis package",
    "fuel_cell": "fuel-cell package",
    "dac_fan_and_contactor": "direct-air-capture fan and contactor package",
    "carbonator": "calcium-looping carbonator",
    "calciner": "calcium-looping calciner",
    "co2_compressor": "carbon-dioxide compressor package",
    "storage_compressor": "gas-storage charging compressor package",
    "storage_expander": "gas-storage discharge expander package",
    "sabatier_reactor": "Sabatier reactor",
    "feed_effluent_heat_exchanger": "Sabatier feed-effluent heat exchanger",
}

SINGLE_SITE_PARAMETER_KEYS = {
    "plant.solar_farm_mw",
    "plant.air_exhaust_hx_effectiveness",
    "strategy.short_strategy",
    "strategy.long_strategy",
    "strategy.f_ocp",
    "strategy.f_socp_long",
    "strategy.parallel_reactor_count",
    "strategy.reactor_scheduling_mode",
    "strategy.product_storage",
    "storage.short_battery.initial_soc_fraction",
    "storage.long_battery.initial_soc_fraction",
    "storage.long_hydrogen.initial_soc_fraction",
    "storage.long_h2_co2.initial_soc_fraction",
}


def load_showcase(path: str | Path = SHOWCASE_PATH) -> pd.DataFrame:
    """Read the showcase site catalogue.

    Only site, lat and lon are used. Every metric column in that file is stale — it
    predates the sizing split, and was produced on synthetic weather with library
    default capacities, battery long storage and no faults. See
    data/showcase/README.md. Do not surface its numbers without regenerating first.
    """
    return pd.read_csv(path)


# Reference cities for labelling cached coordinates that are not showcase sites.
# Only used for display, so a coarse list of well-known places is enough.
REFERENCE_CITIES = (
    ("Tromso", 69.649, 18.955), ("Helsinki", 60.170, 24.938),
    ("Oslo", 59.914, 10.752), ("Stockholm", 59.329, 18.069),
    ("Inverness", 57.478, -4.225), ("Aberdeen", 57.149, -2.099),
    ("Glasgow", 55.864, -4.252), ("Edinburgh", 55.953, -3.188),
    ("Copenhagen", 55.676, 12.568), ("Leeds", 53.801, -1.549),
    ("Hamburg", 53.551, 9.993), ("Manchester", 53.481, -2.243),
    ("Dublin", 53.350, -6.260), ("Sheffield", 53.383, -1.467),
    ("Berlin", 52.520, 13.405), ("Birmingham", 52.487, -1.890),
    ("Amsterdam", 52.370, 4.895), ("Warsaw", 52.230, 21.012),
    ("London", 51.507, -0.128), ("Cardiff", 51.481, -3.179),
    ("Swansea", 51.621, -3.944), ("Leipzig", 51.340, 12.375),
    ("Dresden", 51.050, 13.738), ("Brussels", 50.851, 4.352),
    ("Prague", 50.076, 14.438), ("Krakow", 50.065, 19.945),
    ("Paris", 48.857, 2.352), ("Vienna", 48.208, 16.374),
    ("Munich", 48.135, 11.582), ("Budapest", 47.498, 19.040),
    ("Zurich", 47.377, 8.542), ("Lyon", 45.764, 4.836),
    ("Zagreb", 45.815, 15.982), ("Milan", 45.464, 9.190),
    ("Venice", 45.441, 12.316), ("Verona", 45.438, 10.993),
    ("Turin", 45.071, 7.687), ("Bologna", 44.494, 11.343),
    ("Bucharest", 44.427, 26.103), ("Genoa", 44.407, 8.934),
    ("Bordeaux", 44.838, -0.579), ("Belgrade", 44.787, 20.449),
    ("Florence", 43.770, 11.256), ("Toulouse", 43.605, 1.444),
    ("Marseille", 43.296, 5.370), ("Sofia", 42.698, 23.322),
    ("Rome", 41.903, 12.496), ("Barcelona", 41.385, 2.173),
    ("Porto", 41.158, -8.629), ("Istanbul", 41.008, 28.978),
    ("Madrid", 40.417, -3.704), ("Valencia", 39.470, -0.377),
    ("Lisbon", 38.722, -9.139), ("Athens", 37.984, 23.728),
    ("Seville", 37.389, -5.985), ("Nicosia", 35.186, 33.382),
)
NEAREST_CITY_LIMIT_KM = 150.0


def nearest_city_label(lat: float, lon: float) -> str | None:
    """Name a coordinate after the closest reference city, if one is near enough."""
    scale = math.cos(math.radians(lat))
    best_name, best_km = None, math.inf
    for name, city_lat, city_lon in REFERENCE_CITIES:
        km = 111.0 * math.hypot(lat - city_lat, (lon - city_lon) * scale)
        if km < best_km:
            best_name, best_km = name, km
    return f"near {best_name}" if best_km <= NEAREST_CITY_LIMIT_KM else None


def cached_site_entries():
    """Every cached site, named where it matches the showcase catalogue.

    ``runnable`` marks the sites holding a contiguous span long enough for a
    default run, so the cached-weather button can select exactly those and leave
    partial caches alone rather than queueing runs that are bound to fail.
    """
    required = WeatherConfig(0, 0).training_years + WeatherConfig(0, 0).evaluation_years
    try:
        catalogue = load_showcase()[["site", "lat", "lon"]].to_dict("records")
    except Exception:
        catalogue = []
    entries = []
    for site in cached_weather_sites():
        name = next(
            (row["site"] for row in catalogue
             if abs(float(row["lat"]) - site["lat"]) < 0.01
             and abs(float(row["lon"]) - site["lon"]) < 0.01),
            None,
        )
        years = site["years"]
        contiguous = len(years) == years[-1] - years[0] + 1
        # Selections dedupe on coordinates, not the label, so naming an unmatched
        # site after its closest city stays compatible with clicking its dot.
        fallback = (nearest_city_label(site["lat"], site["lon"])
                    or f"{site['lat']:.2f}, {site['lon']:.2f}")
        entries.append({
            "site": name or fallback,
            "lat": site["lat"], "lon": site["lon"], "years": years,
            "named": name is not None,
            "runnable": contiguous and len(years) >= required,
        })
    return entries


def cached_site_options():
    """Dropdown entries for every site already present in the weather cache."""
    options = []
    for entry in cached_site_entries():
        years = entry["years"]
        label = (f"{entry['site']} — {len(years)} years {years[0]}-{years[-1]}"
                 + ("" if entry["runnable"] else " (incomplete)"))
        options.append({"label": label, "value": json.dumps(
            {"lat": entry["lat"], "lon": entry["lon"], "latest": years[-1]})})
    return options


def _metric_label(label: str, description: str):
    """Radio label with a hover note explaining what the metric means."""
    return html.Span([
        label,
        html.Abbr(" ⓘ", title=description,
                  style={"cursor": "help", "textDecoration": "none",
                         "color": "#4a6b8a", "fontWeight": 700}),
    ])


# (main-page id, twin id) pairs kept in step by _register_showcase_mirrors.
SHOWCASE_MIRRORED_CONTROLS = (
    ("f-ocp", "f-ocp-showcase"),
    ("f-socp", "f-socp-showcase"),
    ("short-strategy", "short-strategy-showcase"),
    ("long-storage", "long-storage-showcase"),
    ("weather-source", "weather-source-showcase"),
    ("fault-seed", "fault-seed-showcase"),
    *((f"fault-{component}-{field}", f"fault-{component}-{field}-showcase")
      for component in ("battery", "hydrogen", "sabatier", "dac")
      for field in ("months", "duration", "capacity")),
)


def _showcase_fault_sliders(component: str, label: str) -> html.Div:
    ranges = (
        ("Mean interval (months)", "months", 0, 120, 1, 6,
         {0: "Off", 1: "1", 6: "6", 24: "24", 60: "60", 120: "120"}),
        ("Mean duration (h)", "duration", 1, 720, 1, 48,
         {1: "1", 48: "48", 168: "168", 360: "360", 720: "720"}),
        ("Mean retained capacity", "capacity", 0, 1, 0.05, 0.8,
         {0: "0%", 0.5: "50%", 0.8: "80%", 1: "100%"}),
    )
    return html.Div([
        html.Strong(label, style={"color": "#173f35"}),
        *(html.Div([
            html.Label(title, style={"fontWeight": 600, "fontSize": "0.82rem"}),
            dcc.Slider(id=f"fault-{component}-{field}-showcase", min=minimum,
                       max=maximum, step=step, value=value, marks=marks,
                       included=False, persistence=True, persistence_type="local",
                       tooltip={"placement": "bottom", "always_visible": False}),
        ]) for title, field, minimum, maximum, step, value, marks in ranges),
    ], style={"display": "grid",
              "gridTemplateColumns": "repeat(auto-fit,minmax(160px,1fr))",
              "alignItems": "center", "gap": "0.6rem", "padding": "0.35rem 0"})


def showcase_weather_controls():
    """Weather source for showcase runs, kept visible rather than collapsed."""
    return html.Div([
        html.Div("Weather source for showcase runs",
                 style={"fontWeight": 650, "color": "#173f35"}),
        html.Div([
            dcc.RadioItems(
                id="weather-source-showcase", value="demo",
                options={"demo": "Offline synthetic", "ninja": "Renewables.ninja",
                         "cached": "Cached only"},
                persistence=True, persistence_type="local", inline=True,
                style={"display": "flex", "flexWrap": "wrap", "gap": "0.2rem 0.9rem"},
            ),
            html.Button("Use cached weather", id="load-cached-weather-showcase",
                        n_clicks=0, style={"marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "0.9rem",
                  "flexWrap": "wrap", "marginTop": "0.35rem"}),
        html.Div(id="cached-weather-status-showcase", role="status",
                 style={"fontSize": "0.78rem", "color": "#40534d",
                        "marginTop": "0.35rem"}),
    ], style={"margin": "0.6rem 0", "padding": "0.7rem 0.9rem", "background": "white",
              "border": "1px solid #dbe0e8", "borderRadius": "8px"})


def showcase_mirror_controls():
    """Editable copies of the main-page run settings, collapsed by default."""
    minimum, maximum, increment, marks = _slider_scale(0, 2, 0.05)
    slider = dict(min=minimum, max=maximum, step=increment, marks=marks,
                  tooltip={"placement": "bottom", "always_visible": True},
                  persistence=True, persistence_type="local")
    return html.Details([
        html.Summary("Run settings (mirrors the main tab)",
                     style={"cursor": "pointer", "fontWeight": 650,
                            "color": "#173f35", "padding": "0.35rem 0"}),
        html.Div([
            _control("Plant sizing reserve · f_OCP",
                     dcc.Slider(id="f-ocp-showcase", value=0.20, **slider)),
            _control("Long-term storage energy · f_SOCP",
                     dcc.Slider(id="f-socp-showcase", value=1.0, **slider)),
            _control("Operating strategy", dcc.RadioItems(
                id="short-strategy-showcase", value="limping", inline=True,
                options=[{"label": value.replace("_", " ").title(), "value": value}
                         for value in OPERATING_STRATEGIES])),
            _control("Long storage", dcc.RadioItems(
                id="long-storage-showcase", value="battery", inline=True,
                options=_long_storage_options())),
        ], className="control-grid"),
        html.P("Short-term storage is always a battery, and showcase runs always "
               "compute all three information/fault variants so every metric is "
               "populated. Faults are therefore always on here: set a subsystem's "
               "mean interval to 0 to disable it, and set all four to 0 to make the "
               "faulted and imperfect variants coincide.",
               style={"fontSize": "0.8rem", "color": "#60716c", "marginTop": "0.5rem"}),
        html.Details([
            html.Summary("Fault rates",
                         style={"cursor": "pointer", "fontWeight": 600,
                                "color": "#173f35", "padding": "0.3rem 0"}),
            html.Label(["Random seed ",
                        dcc.Input(id="fault-seed-showcase", type="number", value=0,
                                  min=0, step=1, persistence=True,
                                  persistence_type="local",
                                  style={"width": "140px", "height": "32px"})],
                       style={"fontWeight": 600}),
            *(_showcase_fault_sliders(component, label) for component, label in (
                ("battery", "Battery"), ("hydrogen", "Hydrogen apparatus"),
                ("sabatier", "Sabatier reactor"), ("dac", "DAC apparatus"))),
        ], style={"marginTop": "0.5rem"}),
    ], style={"margin": "0.6rem 0", "padding": "0.6rem 0.9rem", "background": "white",
              "border": "1px solid #dbe0e8", "borderRadius": "8px"})


def metric_options(advanced: bool):
    meta = {**SIMPLE_METRIC_META, **(ADVANCED_METRIC_META if advanced else {})}
    return [{"label": _metric_label(label, description), "value": key}
            for key, (label, description) in meta.items()]


def _finite(value):
    """Return a float only when the value is a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def expand_showcase_metrics(row):
    """Flatten the per-variant economics into the flat columns the map plots.

    Every case carries perfect, imperfect and imperfect-with-faults economics, and
    each of those dicts already reports its own annual methane, so the three
    production and three cost metrics come straight out of them.
    """
    row = dict(row)
    outputs = row.get("outputs") or {}
    variants = {
        "perfect": outputs.get("economics_perfect") or {},
        "imperfect": outputs.get("economics_imperfect") or {},
        "faults": outputs.get("economics_imperfect_with_faults") or {},
    }
    for suffix, economics in variants.items():
        row[f"ch4_{suffix}"] = _finite(economics.get("average_annual_methane_kg"))
        row[f"lcom_{suffix}"] = _finite(economics.get("lcom_usd_per_kg_ch4"))
    # Faults are forced on for showcase runs, so the faulted variant is the realised
    # case; fall back to imperfect only if an older record predates that.
    realised = row.get("ch4_faults") if row.get("ch4_faults") is not None else row.get("ch4_imperfect")
    ideal = row.get("ch4_perfect")
    row["production_ratio"] = realised / ideal if realised is not None and ideal else None
    # Against the "do nothing" reference: the same plant, no storage, on-off control,
    # carrying the same faults. Absent from records made before baselines were kept.
    baseline = _finite((outputs.get("economics_baseline") or {}).get(
        "average_annual_methane_kg"))
    # Baseline over realised, so this reads the same way round as production_ratio:
    # a fraction of what the plant actually achieves, not a multiple of the reference.
    row["baseline_production_ratio"] = (
        baseline / realised if baseline is not None and realised else None)
    costed = variants["faults"] or variants["imperfect"]
    row.setdefault("methane_storage_capacity_kg",
                   _finite(costed.get("methane_storage_capacity_kg")))
    # A plant sized on the training record can be under-built for the decade that
    # follows. That shows up as a cheap LCOM, so the map has to say why rather than
    # leave the site looking like a bargain. Records made before the split carry no
    # figure and get no note.
    shortfall = row.get("imperfect_deficit_mwh")
    row["deficit_note"] = (
        DEFICIT_NOTE
        if isinstance(shortfall, (int, float)) and not isinstance(shortfall, bool)
        and math.isfinite(shortfall) and shortfall > 1e-9
        else ""
    )
    for key in (*METRIC_LABELS, *LEGACY_SHOWCASE_COLUMNS):
        row.setdefault(key, None)
    return row


def showcase_data(records=None):
    data = []
    for record in records or []:
        data = [row for row in data if row["site"] != record["site"]]
        data.append(expand_showcase_metrics(normalize_showcase_ratio(record)))
    return pd.DataFrame(data, columns=["site", "lat", "lon", "data_status",
                                      "forecast_cost_ratio_basis", "deficit_note",
                                      *LEGACY_SHOWCASE_COLUMNS, *METRIC_LABELS])


def toggle_showcase_location(click_data, selected):
    selected = list(selected or [])
    if not click_data or not click_data.get("points"):
        return selected
    point = click_data["points"][0]
    lat, lon = float(point["lat"]), float(point["lon"])
    site = point.get("hovertext") or f"{lat:.2f}, {lon:.2f}"
    if any(row["lat"] == lat and row["lon"] == lon for row in selected):
        return [row for row in selected if (row["lat"], row["lon"]) != (lat, lon)]
    return selected + [{"site": site, "lat": lat, "lon": lon}]


# Framed so the whole showcase latitude range is visible without panning: Tromso at
# 69.6N sits well inside the top edge, and Seville at 37.4N well inside the bottom.
MAP_CENTRE = {"lat": 54.0, "lon": 10.0}
MAP_ZOOM = 2.4


def build_showcase_map(metric: str = "lcom_usd_per_kg_ch4", records=None,
                       selected=None, clickable=False, show_grid=True):
    data = showcase_data(records)
    metric = metric if metric in METRIC_LABELS else DEFAULT_MAP_METRIC
    if not data.empty:
        # Plotly cannot size a marker from a missing value.
        data = data.assign(**{MAP_SIZE_COLUMN: data[MAP_SIZE_COLUMN].fillna(0.0)})
    figure = px.scatter_map(
        data, lat="lat", lon="lon", color=metric, size=MAP_SIZE_COLUMN,
        hover_name="site", hover_data={"data_status": True, metric: ":.3g", "lat": ":.2f", "lon": ":.2f",
                                     "forecast_cost_ratio_basis": metric == "forecast_cost_ratio",
                                     "deficit_note": True},
        color_continuous_scale="Viridis_r" if "lcom" in metric else "Viridis",
        zoom=MAP_ZOOM, center=MAP_CENTRE, height=620,
        labels={metric: METRIC_LABELS[metric]}, size_max=MAP_RESULT_SIZE_MAX,
    ) if not data.empty else go.Figure()
    if not data.empty:
        # A site with little production would otherwise get a dot smaller than one
        # grid target, which is what makes the grid steal its hover.
        figure.update_traces(marker_sizemin=MAP_RESULT_SIZE_MIN)
        # Express labels every hover row "column=value". The deficit is a sentence
        # rather than a statistic, so it gets its own bold line with no label, below
        # the numbers. Sites with no deficit carry an empty string and show nothing.
        for trace in figure.data:
            template = getattr(trace, "hovertemplate", None) or ""
            if "deficit_note=" in template:
                trace.hovertemplate = re.sub(
                    r"(<br>)?deficit_note=(%\{customdata\[\d+\]\})",
                    r"<br><b>\2</b>", template,
                )
    if clickable and show_grid:
        targets = build_location_picker().data[0]
        figure.add_trace(targets)
        figure.data = (figure.data[-1], *figure.data[:-1])
    if selected:
        figure.add_trace(go.Scattermap(
            lat=[row["lat"] for row in selected], lon=[row["lon"] for row in selected],
            hovertext=[row["site"] for row in selected], mode="markers",
            marker={"size": 48, "color": "#e77621", "opacity": 0.4},
            name="Selected for run", hovertemplate="%{hovertext}<extra>Click to deselect</extra>",
        ))
        # Put selection halos beneath result dots so their metric hovers stay accessible,
        # and beneath the grid too when the grid is present at index 0.
        insertion = 1 if (clickable and show_grid) else 0
        traces = list(figure.data)
        traces.insert(insertion, traces.pop())
        figure.data = tuple(traces)
    figure.update_layout(map_style="open-street-map", uirevision="showcase",
                         map_zoom=MAP_ZOOM, map_center=MAP_CENTRE, height=620,
                         legend={"orientation": "h", "x": 0, "y": -0.04,
                                 "yanchor": "top"},
                         coloraxis_colorbar={"x": 1.02, "y": 0.5, "len": 0.85},
                         margin={"l": 0, "r": 100, "t": 35, "b": 65})
    return figure


def _number_card(label: str, value: str) -> html.Div:
    return html.Div(
        [html.Div(label, style={"fontSize": "0.8rem", "color": "#586174"}),
         html.Div(value, style={"fontSize": "1.35rem", "fontWeight": 650})],
        style={"padding": "0.8rem", "border": "1px solid #dbe0e8", "borderRadius": "8px",
               "background": "white", "minWidth": "170px"},
    )


INVESTIGATION_OPTIONS = {
    "short_name": ("Operating strategy", OPERATING_STRATEGIES),
    "long_method": ("Storage method", LONG_STORAGE_METHODS),
    "reactor_count": ("Parallel trains", tuple(range(1, 7))),
    "reactor_scheduling_mode": ("Reactor scheduling", ("daily_storage_aware", "seasonal")),
}


SHOWCASE_SETTING_FIELDS = (
    ("farm-mw", "Solar farm (MW)"), ("weather-source", "Weather source"),
    ("latest-year", "Latest API weather year"),
    ("information-mode", "Information mode"), ("short-strategy", "Operating strategy"),
    ("long-storage", "Long-term storage"), ("reactor-count", "Parallel trains"),
    ("reactor-scheduling-mode", "Reactor scheduling"),
    ("capacity-setting-mode", "Capacity setting"),
    ("f-ocp", "f_OCP (starting value when calibrating)"),
    ("f-socp", "f_SOCP (starting value when calibrating)"),
    ("air-hx-effectiveness", "Air/exhaust exchanger effectiveness"),
    ("factor-range-min", "Factor range minimum"), ("factor-range-max", "Factor range maximum"),
    ("factor-range-step", "Factor increment"), ("initial-soc", "Initial SOC fraction"),
    ("fault-enabled", "Fault simulations"), ("fault-seed", "Fault seed"),
    *((f"fault-{component}-{field}", f"{component.title()} faults: {label}")
      for component in ("battery", "hydrogen", "sabatier", "dac")
      for field, label in (("months", "mean interval (months)"),
                           ("duration", "mean duration (hours)"),
                           ("capacity", "retained capacity fraction"))),
)


def _long_storage_options():
    """Long-storage choices, with the product-vessel strategy flagged.

    Methane product storage is the only method that buffers the gate flow, and the
    only one whose seasonal store cannot power the plant. It has not been validated
    against the others, so it carries the badge wherever it is offered.
    """
    label = {
        PRODUCT_STORAGE_METHOD: lambda: html.Span([
            LONG_STORAGE_LABELS[PRODUCT_STORAGE_METHOD] + " ",
            _experimental(
                "Experimental: the plant builds no upstream seasonal store and "
                "follows the sun, with a product vessel holding the summer surplus "
                "so the gate flow stays constant. That store is one-way — it cannot "
                "run the plant — so the trade against the other methods is plant "
                "utilisation and thermal cycling, and it has not been validated. "
                "Use H2 + CO2 for results you intend to quote."),
        ]),
    }
    return [{"label": label[value]() if value in label else LONG_STORAGE_LABELS[value],
             "value": value}
            for value in LONG_STORAGE_METHODS]


def showcase_settings_summary(values):
    labels = {"demo": "Offline synthetic", "ninja": "Renewables.ninja",
              "cached": "Cached only (no API calls)",
              "perfect_only": "Perfect only", "comparison": "Perfect vs imperfect",
              "auto": "Auto-calibrate for each location", "manual": "Manual",
              **LONG_STORAGE_LABELS}
    rows = []
    for (key, label), value in zip(SHOWCASE_SETTING_FIELDS, values):
        # Showcase runs override these two so every map metric is populated; report
        # what will actually run rather than the single-site control's value.
        if key == "fault-enabled":
            display = "Always enabled for showcase runs"
        elif key == "information-mode":
            display = "Perfect vs imperfect (forced for showcase runs)"
        elif value is None:
            display = "Not set"
        elif isinstance(value, str):
            display = labels.get(value, value.replace("_", " ").title())
        else:
            display = str(value)
        rows.append(html.Tr([html.Th(label, style={"textAlign": "left", "paddingRight": "1.5rem"}),
                             html.Td(display)]))
    return html.Div([
        html.P("Live settings for the next run, copied from Single site when Run is clicked. "
               "Each selected map location replaces the Single site coordinates. "
               "One base case is run per location; investigation checkboxes do not add cases. "
               "Saved System & economic parameters and the configured API key also apply. "
               "Changing settings does not recalculate existing dots."),
        html.Table(html.Tbody(rows)),
    ])


def investigation_cases(parameters):
    """Return the base case and unique one-factor variations, in display order."""
    base = {key: parameters.get(key, 1 if key == "reactor_count" else "seasonal")
            for key in INVESTIGATION_OPTIONS}
    cases = [dict(base, label="Base case")]
    for key, (label, options) in INVESTIGATION_OPTIONS.items():
        if key in parameters.get("investigate", []):
            for value in options:
                if value != base[key]:
                    display = LONG_STORAGE_LABELS.get(value, str(value).replace("_", " ").title())
                    cases.append({**base, key: value, "label": f"{label}: {display}"})
    return cases


def _experimental(text):
    """Mark an option whose dispatch rules are not yet trusted.

    Tooltips here describe behaviour, never figures: a number baked into a tooltip
    goes stale the next time the model changes and nothing re-derives it.
    """
    return html.Span("EXPERIMENTAL", title=text, tabIndex=0,
                     className="experimental-badge", **{"aria-label": text})


def _info(text):
    return html.Span("\u24d8", title=text, tabIndex=0, className="info-icon",
                     **{"aria-label": text})


def showcase_demo_panel() -> html.Div:
    """Fill the map from a shipped showcase without running ten cases."""
    return html.Div([
        html.Div([
            html.Strong("Explore results now", className="demo-bar-label"),
            html.Span(
                "Load ten European sites already run on cached Renewables.ninja "
                "reanalysis weather, from the Arctic at Tromsø to Seville. The two "
                "showcases differ only in reactor parallelisation, so loading each in "
                "turn shows what running several smaller trains buys.",
                id="showcase-demo-copy", className="demo-bar-copy",
            ),
        ], className="demo-bar-text"),
        html.Div([
            html.Div([
                html.Button("Load parallel reactor showcase",
                            id="load-demo-showcase-parallel", n_clicks=0,
                            className="demo-bar-button"),
                html.Button("Load single reactor showcase",
                            id="load-demo-showcase-single", n_clicks=0,
                            className="demo-bar-button demo-bar-button-secondary"),
            ], style={"display": "flex", "gap": "0.6rem", "flexWrap": "wrap"}),
            html.Div("", id="showcase-demo-progress", role="status",
                     className="demo-bar-progress"),
            html.Div("", id="showcase-demo-status", role="status",
                     className="demo-bar-status"),
        ], className="demo-bar-action"),
    ], className="demo-bar demo-bar-inset")


def demo_panel() -> html.Div:
    """One-click entry into the shipped results, for a first-time visitor."""
    return html.Div([
        html.Div([
            html.Strong("Start here", className="demo-bar-label"),
            html.Span(
                "Load a complete set of pre-computed results — ten European sites "
                "on real reanalysis weather, plus one worked site with full hourly "
                "dispatch. No simulation run, no weather API key needed.",
                id="demo-bar-copy", className="demo-bar-copy",
            ),
        ], className="demo-bar-text"),
        html.Div([
            html.Button("Load demo case", id="load-demo", n_clicks=0,
                        className="demo-bar-button"),
            html.Div("", id="demo-progress", role="status",
                     className="demo-bar-progress"),
            html.Div("", id="demo-status", role="status", className="demo-bar-status"),
        ], className="demo-bar-action"),
    ], className="demo-bar")


def _section(title, children, colour="#176b55", opened=False, help_text=None):
    return html.Details([
        html.Summary([title, _info(help_text)] if help_text else title),
        html.Div(children, className="section-body"),
    ], open=opened, className="dashboard-section", style={"--section-colour": colour})


def _control(label: str, component) -> html.Div:
    return html.Div(
        [html.Label(label, style={"minHeight": "2.5rem", "display": "flex",
                                  "alignItems": "flex-end", "fontWeight": 600}),
         component,
         *([dcc.Checklist(id={"type": "investigate", "parameter": key},
              options=[{"label": "Include in test cases", "value": key}], value=[],
              className="investigation-toggle")] if (key := {
                  "short-strategy": "short_name", "long-storage": "long_method",
                  "reactor-count": "reactor_count",
                  "reactor-scheduling-mode": "reactor_scheduling_mode",
              }.get(getattr(component, "id", None))) else [])],
        style={"display": "flex", "flexDirection": "column", "gap": "0.35rem",
               "minWidth": 0},
    )


def _resolve_ninja_api_key(entered_key: str | None) -> str | None:
    """Prefer the configured environment token, then an in-app token."""
    configured_key = _load_api_token()
    if configured_key:
        return configured_key
    if isinstance(entered_key, str):
        return entered_key.strip() or None
    return None


def _slider_scale(minimum, maximum, increment) -> tuple[float, float, float, dict]:
    minimum, maximum, increment = float(minimum), float(maximum), float(increment)
    if not all(math.isfinite(value) for value in (minimum, maximum, increment)):
        raise ValueError("slider scale values must be finite")
    if minimum < 0:
        raise ValueError("the minimum cannot be negative")
    if maximum <= minimum:
        raise ValueError("the maximum must be greater than the minimum")
    if increment <= 0 or increment > maximum - minimum:
        raise ValueError("the increment must be positive and no larger than the range")
    span = maximum - minimum
    ticks = {minimum, maximum}
    for index in range(1, 4):
        aligned = minimum + round((span * index / 4) / increment) * increment
        ticks.add(min(maximum, max(minimum, aligned)))
    marks = {tick: f"{tick:g}" for tick in sorted(ticks)}
    return minimum, maximum, increment, marks


def _slow_calibration_warning(reactor_scheduling_mode: str,
                              capacity_setting_mode: str) -> str:
    if (reactor_scheduling_mode == "daily_storage_aware"
            and capacity_setting_mode == "auto"):
        return (
            "Warning: perfect daily reactor reallocation with auto-calibrated "
            "overcapacities will be very slow because every optimiser candidate "
            "requires an additional chronological reactor-commitment planning pass."
        )
    return ""


CALIBRATION_BLOCK_STYLE = {
    "gridColumn": "1 / -1", "display": "grid", "gap": "0.5rem",
    "padding": "0.7rem 0.9rem", "background": "#f3f0e4",
    "border": "1px solid #ddd0a8", "borderRadius": "8px",
}


def _factor_slider(label: str, component_id: str, value: float,
                   description: str, scale: tuple[float, float, float] = (0, 2, 0.05)) -> html.Div:
    minimum, maximum, increment, marks = _slider_scale(*scale)
    return html.Div(
        [
            html.Div(label, style={"fontWeight": 650, "color": "#173f35"}),
            html.Div(description, style={"fontSize": "0.78rem", "color": "#60716c",
                                         "minHeight": "2rem", "lineHeight": 1.3}),
            dcc.Slider(
                id=component_id, min=minimum, max=maximum, step=increment, value=value,
                marks=marks, tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type="local",
            ),
        ],
        style={"padding": "0.85rem 1rem 1.25rem", "background": "white",
               "border": "1px solid #d8e3df", "borderRadius": "10px",
               "boxShadow": "0 1px 3px rgba(23, 107, 85, 0.06)", "minWidth": 0},
    )


def _format_assumption_value(value) -> str:
    if isinstance(value, float):
        if value == float("inf"):
            return "Unbounded"
        return f"{value:.6g}"
    return str(value).replace("_", " ")


def _dataclass_assumption_rows(category: str, prefix: str, instance, metadata: dict) -> list[dict]:
    return [
        {
            "key": f"{prefix}.{name}",
            "category": category,
            "parameter": name,
            "value": _format_assumption_value(getattr(instance, name)),
            "raw_value": getattr(instance, name),
            "unit": unit,
            "description": description,
            "editable": True,
        }
        for name, (unit, description) in metadata.items()
    ]


def build_assumption_rows(costs_path: str | Path = "costs.json") -> list[dict]:
    """Return the model defaults and all numerical economic assumptions for display."""
    rows = []
    rows.extend(_dataclass_assumption_rows(
        "Process plant", "plant", PlantParameters(), PLANT_ASSUMPTION_META
    ))
    rows.extend(_dataclass_assumption_rows(
        "Physical properties", "plant", PlantParameters(),
        PHYSICAL_PROPERTY_ASSUMPTION_META,
    ))
    rows.extend(_dataclass_assumption_rows(
        "Thermal dynamics", "thermal", ThermalParameters(), THERMAL_ASSUMPTION_META
    ))
    rows.extend(_dataclass_assumption_rows(
        "Operating strategy", "strategy", StrategyConfig(), STRATEGY_ASSUMPTION_META
    ))
    weather = WeatherConfig(51.5074, -0.1278)
    rows.extend(_dataclass_assumption_rows(
        "Weather (API)", "weather", weather, WEATHER_ASSUMPTION_META
    ))
    rows.extend(_dataclass_assumption_rows(
        "Weather (synthetic)", "synthetic", SyntheticWeatherParameters(),
        SYNTHETIC_WEATHER_ASSUMPTION_META,
    ))
    rows.extend([
        {"key": "synthetic.start_year", "category": "Weather (synthetic)",
         "parameter": "start_year", "value": "2010", "raw_value": 2010, "unit": "year",
         "description": "First UTC year generated for the deterministic offline profile.",
         "editable": True},
        {"key": "synthetic.training_years", "category": "Weather (synthetic)",
         "parameter": "training_years", "value": "5", "raw_value": 5, "unit": "years",
         "description": "Number of offline years used to build the climatology forecast.",
         "editable": True},
        {"key": "synthetic.evaluation_years", "category": "Weather (synthetic)",
         "parameter": "evaluation_years", "value": "1", "raw_value": 1, "unit": "years",
         "description": "Number of final offline years used for model evaluation. Every "
                        "hour is kept for the dispatch plots, so a long period makes a "
                        "large saved file — roughly 10 MB per evaluation year per "
                        "result category.",
         "editable": True},
        {"key": "synthetic.seed_latitude_multiplier", "category": "Weather (synthetic)",
         "parameter": "seed_latitude_multiplier", "value": "100", "raw_value": 100.0,
         "unit": "seed/degree", "description": "Latitude multiplier in the deterministic cloud random seed.",
         "editable": True},
        {"key": "synthetic.seed_longitude_multiplier", "category": "Weather (synthetic)",
         "parameter": "seed_longitude_multiplier", "value": "10", "raw_value": 10.0,
         "unit": "seed/degree", "description": "Longitude multiplier in the deterministic cloud random seed.",
         "editable": True},
    ])
    rows.extend([
        {"key": None, "category": "Scenario controls", "parameter": "latitude",
         "value": "51.5074", "unit": "degrees north",
         "description": "Site latitude selected on the Single site tab.", "editable": False},
        {"key": None, "category": "Scenario controls", "parameter": "longitude",
         "value": "-0.1278", "unit": "degrees east",
         "description": "Site longitude selected on the Single site tab.", "editable": False},
        {"key": None, "category": "Scenario controls", "parameter": "weather_source",
         "value": "Offline synthetic", "unit": "—",
         "description": "Weather source selected on the Single site tab.", "editable": False},
        {"key": None, "category": "Scenario controls", "parameter": "latest_api_year",
         "value": "2025", "unit": "year",
         "description": "Latest API evaluation year selected on the Single site tab.", "editable": False},
        {"key": None, "category": "Scenario controls", "parameter": "information_mode",
         "value": "Perfect vs imperfect", "unit": "—",
         "description": "Forecast-information comparison mode selected on the Single site tab.", "editable": False},
        {"key": None, "category": "Scenario controls", "parameter": "capacity_setting_mode",
         "value": "Set capacities manually", "unit": "—",
         "description": "Selects manual capacity factors or perfect-information LCOM auto-calibration on the Single site tab.", "editable": False},
        {"key": None, "category": "Scenario controls", "parameter": "long_storage_method",
         "value": "Battery", "unit": "—",
         "description": "Long-duration storage carrier selected on the Single site tab.", "editable": False},
        {"key": None, "category": "Scenario controls", "parameter": "stochastic_fault_scenario",
         "value": "Disabled", "unit": "—",
         "description": "Optional seeded subsystem fault distributions selected on the Single site tab.", "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "simulation_timestep",
         "value": "1", "unit": "h",
         "description": "Dispatch and storage state updates operate on hourly UTC weather samples.", "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "short_storage_cycle",
         "value": "1", "unit": "calendar day",
         "description": "Short storage balances each UTC calendar day before passing residual energy to long storage.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "annualisation_basis",
         "value": "8760", "unit": "h/year",
         "description": "Totals are annualised using 8,760 hours per representative year.", "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "forecast_method",
         "value": "Hourly climatology", "unit": "—",
         "description": "Imperfect forecasts average training data by month, day, and UTC hour.", "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "capacity_calibration_limit",
         "value": str(MAX_CAPACITY_CALIBRATION_EVALUATIONS), "unit": "perfect-information evaluations",
         "description": "Hard upper bound on unique f_OCP and f_SOCP pairs evaluated during automatic LCOM calibration.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "capacity_calibration_objective",
         "value": "Perfect-information LCOM", "unit": "USD/kg CH4",
         "description": "Automatic calibration minimizes perfect-information LCOM and excludes candidates with a cyclic energy deficit.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "thermophysical_source",
         "value": "CoolProp with fallback", "unit": "—",
         "description": "Gas properties use CoolProp when available and the editable constant-property fallbacks otherwise.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "co2_compressor_inlet_pressure",
         "value": "Calculated equilibrium pressure", "unit": "bar",
         "description": "The compressor inlet is calculated from calcination enthalpy entropy and temperature using the van't Hoff relation.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "sabatier_heat_route",
         "value": "Feed heating then rejection", "unit": "—",
         "description": "Sabatier reaction heat offsets CO2 and H2 feed heating directly and any surplus is discarded.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "hydrogen_discharge_priority",
         "value": "Sabatier then fuel cell", "unit": "—",
         "description": "Stored hydrogen first replaces new electrolysis for Sabatier feed and remaining hydrogen may supply the electrical bus through the fuel cell.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "h2_co2_storage_ratio",
         "value": "Calculated Sabatier stoichiometry", "unit": "kg CO2/kg H2",
         "description": "The paired-gas strategy charges CO2 with at least the stoichiometric hydrogen quantity and may add extra hydrogen after the CO2 vessel is full.",
         "editable": False},
        {"key": None, "category": "Fixed model structure", "parameter": "stored_co2_energy_scope",
         "value": "DAC through compression", "unit": "—",
         "description": "Charging stored CO2 includes DAC fan calcination solids and air heating losses and compression; discharging it avoids those same upstream duties.",
         "editable": False},
    ])

    storage_profiles = (
        ("Short storage — battery", StorageParameters(
            method="battery", initial_soc_fraction=0.50,
        )),
        ("Long storage — battery", StorageParameters(
            self_discharge_fraction_per_h=1e-5, initial_soc_fraction=0.50,
        )),
        ("Long storage — hydrogen", StorageParameters(
            method="hydrogen",
            self_discharge_fraction_per_h=1e-6, initial_soc_fraction=0.50,
        )),
        ("Long storage — H2 + CO2 gas", StorageParameters(
            method="h2_co2", self_discharge_fraction_per_h=1e-6,
            co2_self_discharge_fraction_per_h=1e-6, initial_soc_fraction=0.50,
        )),
    )
    for category, storage in storage_profiles:
        prefix = {
            "Short storage — battery": "storage.short_battery",
            "Long storage — battery": "storage.long_battery",
            "Long storage — hydrogen": "storage.long_hydrogen",
            "Long storage — H2 + CO2 gas": "storage.long_h2_co2",
        }[category]
        rows.extend(_dataclass_assumption_rows(
            category, prefix, storage, STORAGE_ASSUMPTION_META
        ))

    rows.extend([
        {"key": None, "category": "Sizing and faults", "parameter": "storage_energy_capacity",
         "value": "Short exactly sized; long × f_SOCP", "unit": "kWh carrier energy",
         "description": "Short-term storage equals its calculated daily-cycle requirement while f_SOCP scales only the calculated long-term residual-cycle requirement.",
         "editable": False},
        {"key": None, "category": "Sizing and faults", "parameter": "storage_charge_discharge_power",
         "value": "Observed unconstrained peak", "unit": "kW",
         "description": "Storage transfer power is not throttled; the peak charge and discharge actually used are reported after the run and used for costing.",
         "editable": False},
        {"key": None, "category": "Sizing and faults", "parameter": "stochastic_fault_controls",
         "value": "Single site tab", "unit": "—",
         "description": "Seeded battery, hydrogen, Sabatier, and DAC fault distributions are configured per run.",
         "editable": False},
    ])

    with Path(costs_path).open(encoding="utf-8") as stream:
        costs = json.load(stream)
    for path, (unit, description) in ECONOMIC_ASSUMPTION_META.items():
        section, key = path.split(".", 1)
        rows.append({
            "key": f"economic.{path}",
            "category": "Economics — " + section.replace("_", " ").title(),
            "parameter": key,
            "value": _format_assumption_value(costs[section][key]),
            "raw_value": costs[section][key],
            "unit": unit,
            "description": description,
            "editable": True,
        })
    for unit_name, unit_costs in costs["plant_units"].items():
        equipment = PLANT_UNIT_DESCRIPTIONS[unit_name]
        unit_metadata = {
            "reference_capacity_kw": (
                "kW", f"Reference capacity used to scale installed cost for the {equipment}."
            ),
            "reference_capacity_kg_ch4_h": (
                "kg CH4/h", f"Reference methane throughput used to scale installed cost for the {equipment}."
            ),
            "reference_installed_cost_usd": (
                "USD", f"Installed cost of the {equipment} at its reference capacity."
            ),
            "scaling_exponent": (
                "dimensionless", f"Capacity-scaling exponent applied to the {equipment} cost correlation."
            ),
        }
        for key, value in unit_costs.items():
            unit, description = unit_metadata[key]
            rows.append({
                "key": f"economic.plant_units.{unit_name}.{key}",
                "category": "Economics — Process units",
                "parameter": f"{unit_name}.{key}",
                "value": _format_assumption_value(value),
                "raw_value": value,
                "unit": unit,
                "description": description,
                "editable": True,
            })
    for row in rows:
        if row.get("key") in SINGLE_SITE_PARAMETER_KEYS:
            row["editable"] = False
            row["description"] += " Set this scenario value on the Single site tab."
    return rows


def default_assumption_config() -> dict:
    return {
        row["key"]: row["raw_value"]
        for row in build_assumption_rows()
        if row.get("editable")
    }


def _assumption_section(config: dict, prefix: str, defaults, metadata: dict) -> dict:
    values = {}
    for name in metadata:
        default = getattr(defaults, name)
        value = config.get(f"{prefix}.{name}", default)
        if value is None:
            raise ValueError(f"Assumption {prefix}.{name} cannot be blank")
        if isinstance(default, int) and not isinstance(default, bool):
            value = int(value)
        elif isinstance(default, float):
            value = float(value)
        else:
            value = str(value)
        values[name] = value
    return values


def _economic_costs_from_assumptions(config: dict,
                                      costs_path: str | Path = "costs.json") -> dict:
    with Path(costs_path).open(encoding="utf-8") as stream:
        costs = json.load(stream)
    for key, value in config.items():
        if not key.startswith("economic."):
            continue
        path = key.removeprefix("economic.").split(".")
        target = costs
        try:
            for part in path[:-1]:
                target = target[part]
        except KeyError:
            # Ignore keys retained in browser storage from an older cost schema.
            continue
        if path[-1] not in target:
            continue
        original = target[path[-1]]
        if value is None:
            raise ValueError(f"Assumption {key} cannot be blank")
        target[path[-1]] = (
            int(value) if isinstance(original, int) and not isinstance(original, bool)
            else float(value) if isinstance(original, float)
            else str(value)
        )
    return costs


def build_assumptions_table() -> html.Div:
    rows = build_assumption_rows()
    headings = ("Category", "Parameter", "Value", "Unit", "Description")
    header = html.Tr([
        html.Th(label, style={"padding": "0.7rem", "textAlign": "left",
                              "borderBottom": "2px solid #aac2ba", "position": "sticky",
                              "top": 0, "background": "#edf5f2", "zIndex": 1})
        for label in headings
    ])
    input_style = {"width": "130px", "padding": "0.38rem 0.45rem",
                   "border": "1px solid #b9c9c3", "borderRadius": "5px",
                   "boxSizing": "border-box"}
    body = [
        html.Tr([
            html.Td(
                dcc.Input(
                    id={"type": "assumption-input", "key": row["key"]},
                    type="number" if isinstance(row.get("raw_value"), (int, float)) else "text",
                    value=row.get("raw_value"), debounce=True,
                    step=(1 if isinstance(row.get("raw_value"), int) else "any"),
                    style=input_style,
                ) if key == "value" and row["editable"] else row[key],
                style={"padding": "0.58rem 0.7rem", "verticalAlign": "top",
                                     "borderBottom": "1px solid #e2e8e5",
                                     "whiteSpace": "nowrap" if key != "description" else "normal"}
            )
            for key in ("category", "parameter", "value", "unit", "description")
        ], style={"background": "#f7faf9" if index % 2 else "white"})
        for index, row in enumerate(rows)
    ]
    return html.Div(
        html.Table([html.Thead(header), html.Tbody(body)],
                   style={"width": "100%", "borderCollapse": "collapse",
                          "fontSize": "0.9rem"}),
        id="assumptions-table",
        style={"overflow": "auto", "maxHeight": "72vh", "border": "1px solid #d8e3df",
               "borderRadius": "10px", "background": "white"},
    )


def empty_dispatch_figure(message: str = "Run a single-site case to display dispatch results."):
    figure = go.Figure()
    figure.add_annotation(text=message,
                          x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                          font={"size": 16, "color": "#586174"})
    figure.update_layout(template="plotly_white", height=420,
                         xaxis={"visible": False}, yaxis={"visible": False},
                         margin={"l": 30, "r": 30, "t": 40, "b": 30})
    return figure


def empty_reactor_count_figure(message: str = "Run a single-site case to display reactor counts."):
    return empty_dispatch_figure(message)


LIMIT_COLOURS = {
    "At target": "#dfe7e3",
    "Solar resource": "#e8b44a",
    "Plant capacity": "#5b8db8",
    "DAC fault": "#a2423d",
    "Hydrogen fault": "#7b5aa6",
    "Sabatier fault": "#d2691e",
    "Battery fault": "#2e8b8b",
}
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


CITATIONS = (
    {
        "what": "CoolProp",
        "used_for": "Real-fluid enthalpy, entropy and density for CO2, H2 and CH4 "
                    "in the compression, expansion and sensible-heating duties. "
                    "Ideal-gas correlations are used as a fallback where a state "
                    "lies outside the library's range.",
        "reference": "Bell, I. H., Wronski, J., Quoilin, S. and Lemort, V. (2014). "
                     "Pure and Pseudo-pure Fluid Thermophysical Property Evaluation "
                     "and the Open-Source Thermophysical Property Library CoolProp. "
                     "Industrial & Engineering Chemistry Research, 53(6), 2498-2508.",
        "doi": "https://doi.org/10.1021/ie4033999",
    },
    {
        "what": "Renewables.ninja (solar PV)",
        "used_for": "Hourly photovoltaic capacity factors and ambient temperature "
                    "for every real-weather site, retrieved per year and cached "
                    "locally.",
        "reference": "Pfenninger, S. and Staffell, I. (2016). Long-term patterns of "
                     "European PV output using 30 years of validated hourly "
                     "reanalysis and satellite data. Energy, 114, 1251-1265.",
        "doi": "https://doi.org/10.1016/j.energy.2016.08.060",
    },
    {
        "what": "Renewables.ninja (companion paper)",
        "used_for": "Cited alongside the solar paper as the platform asks, and the "
                    "source of the bias-correction approach the service applies.",
        "reference": "Staffell, I. and Pfenninger, S. (2016). Using bias-corrected "
                     "reanalysis to simulate current and future wind power output. "
                     "Energy, 114, 1224-1239.",
        "doi": "https://doi.org/10.1016/j.energy.2016.08.068",
    },
    {
        "what": "MERRA-2 reanalysis",
        "used_for": "The underlying reanalysis dataset behind the Renewables.ninja "
                    "series used here; the app requests the 'merra2' dataset by "
                    "default.",
        "reference": "Gelaro, R. et al. (2017). The Modern-Era Retrospective "
                     "Analysis for Research and Applications, Version 2 (MERRA-2). "
                     "Journal of Climate, 30(14), 5419-5454.",
        "doi": "https://doi.org/10.1175/JCLI-D-16-0758.1",
    },
)


def citations_panel():
    """Sources the model depends on, with what each one is actually used for."""
    entries = []
    for item in CITATIONS:
        entries.append(html.Div([
            html.H3(item["what"], style={"margin": "0 0 0.3rem", "color": "#173f35"}),
            html.P(item["used_for"], style={"margin": "0 0 0.5rem",
                                            "color": "#40534d"}),
            html.Blockquote(
                item["reference"],
                style={"margin": "0 0 0.4rem", "padding": "0.55rem 0.9rem",
                       "borderLeft": "4px solid #176b55", "background": "#f4f8f6",
                       "fontStyle": "italic"},
            ),
            html.A(item["doi"], href=item["doi"], target="_blank",
                   rel="noopener noreferrer", style={"fontSize": "0.86rem"}),
        ], style={"padding": "0.9rem 0", "borderBottom": "1px solid #e5eeeb"}))
    return html.Div([
        html.H2("Citations", style={"marginBottom": "0.35rem"}),
        html.P("Third-party data and libraries this model depends on. Cost figures "
               "are not taken from any of these: they are illustrative placeholders "
               "and are flagged as such wherever they appear.",
               style={"maxWidth": "900px", "color": "#40534d"}),
        *entries,
    ], style={"padding": "0.6rem 0"})


def empty_limiting_figure(message: str = "Run a case, then load its plots."):
    figure = go.Figure()
    figure.update_layout(template="plotly_white", height=330,
                         annotations=[{"text": message, "showarrow": False,
                                       "font": {"size": 14}}],
                         xaxis={"visible": False}, yaxis={"visible": False})
    return figure


def build_limiting_subsystem_figure(simulation, title: str = ""):
    """Calendar of what held production back on each day of the evaluation."""
    daily = daily_limiting_subsystem(simulation.hourly)
    if daily.empty:
        return empty_limiting_figure("No dispatch hours to classify.")
    present = [name for name in LIMIT_CATEGORIES if name in set(daily)]
    codes = {name: number for number, name in enumerate(present)}
    years = sorted({day.year for day in daily.index})
    figure = go.Figure()
    for row, year in enumerate(years):
        grid = np.full((12, 31), np.nan)
        text = np.full((12, 31), "", dtype=object)
        for day, name in daily.items():
            if day.year == year:
                grid[day.month - 1, day.day - 1] = codes[name]
                text[day.month - 1][day.day - 1] = f"{day:%d %b %Y}<br>{name}"
        figure.add_trace(go.Heatmap(
            z=grid, text=text, hovertemplate="%{text}<extra></extra>",
            x=list(range(1, 32)), y=list(MONTH_NAMES), showscale=False,
            xgap=1, ygap=1, zmin=-0.5, zmax=max(len(present) - 0.5, 0.5),
            colorscale=_discrete_colourscale([LIMIT_COLOURS[n] for n in present]),
            visible=row == 0, name=str(year),
        ))
    counts = daily.value_counts()
    # A legend built from dummy traces, because a heatmap has no per-category entry.
    for name in present:
        figure.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=f"{name} ({counts[name]} days)",
            marker={"size": 12, "symbol": "square", "color": LIMIT_COLOURS[name],
                    "line": {"color": "#98a6a1", "width": 1}},
        ))
    if len(years) > 1:
        figure.update_layout(updatemenus=[{
            "buttons": [{"label": str(year), "method": "update",
                         "args": [{"visible": [i == row for i in range(len(years))]
                                   + [True] * len(present)}]}
                        for row, year in enumerate(years)],
            "direction": "down", "showactive": True, "x": 1.0, "xanchor": "right",
            "y": 1.18, "yanchor": "top",
        }])
    figure.update_yaxes(autorange="reversed", title_text="")
    figure.update_xaxes(title_text="Day of month", dtick=2)
    figure.update_layout(
        title=title or "Limiting subsystem by day",
        template="plotly_white", height=380,
        margin={"l": 60, "r": 25, "t": 70, "b": 55},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18,
                "xanchor": "left", "x": 0},
    )
    return figure


def _discrete_colourscale(colours):
    """Flat bands so heatmap codes map to exactly one colour each."""
    steps = len(colours)
    scale = []
    for index, colour in enumerate(colours):
        scale.append([index / steps, colour])
        scale.append([(index + 1) / steps, colour])
    return scale


def empty_comparison_figure(message: str = "Include categories in test cases, then investigate to compare results."):
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font={"size": 15, "color": "#586174"})
    figure.update_layout(template="plotly_white", height=430,
                         xaxis={"visible": False}, yaxis={"visible": False},
                         margin={"l": 35, "r": 25, "t": 45, "b": 35})
    return figure


def build_active_reactor_figure(simulation, title: str = "Reactors active by day"):
    """Plot each day's maximum planned and realized integer reactor counts."""
    required = {"planned_reactor_count", "actual_reactor_count"}
    if simulation is None or simulation.hourly.empty or not required.issubset(simulation.hourly):
        figure = go.Figure()
        figure.add_annotation(
            text="Run a case to display daily reactor counts.", x=0.5, y=0.5,
            xref="paper", yref="paper", showarrow=False,
        )
    else:
        daily = simulation.hourly[
            ["planned_reactor_count", "actual_reactor_count"]
        ].resample("D").max()
        figure = go.Figure()
        seasonal_summary = seasonal_reactor_count_summary(simulation)
        planned_name = (
            f"Forecast-planned trains ({seasonal_summary})"
            if seasonal_summary else "Forecast-planned trains"
        )
        figure.add_trace(go.Scatter(
            x=daily.index, y=daily["planned_reactor_count"],
            mode="lines", line={"shape": "hv", "dash": "dash", "width": 2},
            name=planned_name,
            hovertemplate="%{x|%Y-%m-%d}<br>Planned: %{y:.0f}<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=daily.index, y=daily["actual_reactor_count"],
            mode="lines+markers", line={"shape": "hv", "width": 2},
            marker={"size": 4}, name="Actual active trains",
            hovertemplate="%{x|%Y-%m-%d}<br>Active: %{y:.0f}<extra></extra>",
        ))
        maximum = max(1.0, float(daily.to_numpy().max(initial=0.0)))
        figure.update_yaxes(range=[-0.1, maximum + 0.5], dtick=1)
    figure.update_layout(
        title=title, template="plotly_white", height=430,
        xaxis_title="UTC day", yaxis_title="Maximum reactors active during day",
        hovermode="x unified", margin={"l": 70, "r": 25, "t": 65, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01,
                "xanchor": "left", "x": 0},
    )
    return figure


def _case_metric_value(result, metric: str, *, perfect: bool | None = None,
                       category: str | None = None) -> float:
    if category is None:
        category = "perfect" if perfect else "imperfect"
    simulation, economics, _ = _result_category_parts(result, category)
    if simulation is None or economics is None:
        return float("nan")
    if metric in {
        "lcom_usd_per_kg_ch4", "average_annual_methane_kg",
        "total_capex_usd", "annual_opex_usd_per_year",
        "methane_storage_capex_usd",
    }:
        value = economics.get(metric)
    elif metric == "plant_utilisation_percent":
        value = 100.0 * float(simulation.metrics["plant_utilisation"])
    elif metric == "curtailment_percent":
        value = 100.0 * float(simulation.metrics["curtailment_fraction"])
    elif metric == "average_annual_methane_shortfall_kg":
        value = simulation.metrics[metric]
    elif metric == "annual_balance_deficit_mwh":
        value = result.sizing.average_annual_balance_deficit_kwh / 1000.0
    elif metric == "forced_shutdown_hours":
        value = simulation.metrics[metric]
    elif metric == "short_storage_capacity_mwh":
        value = simulation.metadata["short_storage"]["capacity_kwh"] / 1000.0
    elif metric == "long_storage_capacity_mwh":
        value = simulation.metadata["long_storage"]["capacity_kwh"] / 1000.0
    elif metric == "methane_storage_capacity_kg":
        value = simulation.metadata["methane_storage"]["capacity_kg"]
    elif metric == "short_storage_power_mw":
        power = simulation.metadata["storage_power"]
        value = max(power["short_installed_charge_kw"],
                    power["short_installed_discharge_kw"]) / 1000.0
    elif metric == "long_storage_power_mw":
        power = simulation.metadata["storage_power"]
        value = max(power["long_installed_charge_kw"],
                    power["long_installed_discharge_kw"]) / 1000.0
    else:
        value = (
            1.0 if category == "perfect" and economics.get("lcom_usd_per_kg_ch4") is not None
            else (economics.get("relative_combined_cost_ratio")
                  if category == "imperfect_with_faults"
                  else economics.get("relative_prediction_cost_ratio"))
        )
    return float(value) if value is not None else float("nan")


def build_case_comparison_figure(
    job: dict,
    metric: str,
    selected_storage: tuple[str, ...] | list[str] | None = None,
    selected_strategies: tuple[str, ...] | list[str] | None = None,
    selected_cases: tuple[str, ...] | list[str] | None = None,
):
    """Compare fault-free/faulted imperfect bars with perfect-information points."""
    if metric not in CASE_COMPARISON_METRICS:
        metric = "lcom_usd_per_kg_ch4"
    if not job.get("results"):
        return empty_comparison_figure()
    selected_storage = set(LONG_STORAGE_METHODS if selected_storage is None else selected_storage)
    selected_strategies = set(
        OPERATING_STRATEGIES if selected_strategies is None else selected_strategies
    )
    selected_cases = set(job["results"] if selected_cases is None else selected_cases)
    labels, imperfect_values, faulted_values, perfect_values, colours = [], [], [], [], []
    baseline_values = []
    has_faulted = False
    has_baseline = False
    for case_key in job["results"]:
        short_name, long_method = case_key.split("|")[:2]
        if (short_name not in selected_strategies or
                long_method not in selected_storage or case_key not in selected_cases):
            continue
        result = job["results"].get(case_key)
        if result is None:
            continue
        labels.append(
            job.get("case_labels", {}).get(case_key,
                f"{short_name.replace('_', ' ').title()}<br>{LONG_STORAGE_LABELS[long_method]}")
        )
        imperfect_values.append(_case_metric_value(result, metric, category="imperfect"))
        if getattr(result, "imperfect_with_faults", None) is not None:
            faulted_values.append(_case_metric_value(
                result, metric, category="imperfect_with_faults",
            ))
            has_faulted = True
        else:
            faulted_values.append(float("nan"))
        perfect_values.append(_case_metric_value(result, metric, perfect=True))
        if getattr(result, "baseline", None) is not None:
            baseline_values.append(_case_metric_value(result, metric,
                                                      category="baseline"))
            has_baseline = True
        else:
            baseline_values.append(float("nan"))
        colours.append(CASE_COLOURS[_result_key(short_name, long_method)])
    if not labels:
        return empty_comparison_figure("Select at least one case to compare.")
    perfect_only = job.get("information_mode") == "perfect_only"
    bar_values = perfect_values if perfect_only else imperfect_values
    figure = go.Figure(go.Bar(
        x=labels, y=bar_values, marker_color=colours,
        name="Perfect information" if perfect_only else "Imperfect forecast, fault-free",
        text=["undefined" if pd.isna(value) else f"{value:,.3g}"
              for value in bar_values],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:,.5g}<extra></extra>",
    ))
    if has_faulted:
        figure.add_trace(go.Bar(
            x=labels, y=faulted_values,
            marker={"color": colours, "pattern": {"shape": "/"}},
            opacity=0.72, name="Imperfect with faults",
            text=["undefined" if pd.isna(value) else f"{value:,.3g}"
                  for value in faulted_values],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>Faulted imperfect: %{y:,.5g}<extra></extra>",
        ))
    figure.add_trace(go.Scatter(
        x=labels, y=perfect_values, mode="markers",
        name="Perfect information, no faults",
        marker={"symbol": "diamond", "size": 13, "color": "#111827",
                "line": {"color": "white", "width": 1.5}},
        hovertemplate="%{x}<br>Perfect: %{y:,.5g}<extra></extra>",
    ))
    if has_baseline:
        # The naive reference: same sized plant, no storage, no scheduling.
        figure.add_trace(go.Scatter(
            x=labels, y=baseline_values, mode="markers",
            name="Baseline: no storage, run when sunny",
            marker={"symbol": "x-thin", "size": 13, "color": "#a2423d",
                    "line": {"color": "#a2423d", "width": 3}},
            hovertemplate="%{x}<br>Baseline: %{y:,.5g}<extra></extra>",
        ))
    if job.get("run_scope") == "investigate":
        figure.update_xaxes(tickmode="array", tickvals=labels,
                            ticktext=[label.split(" · ")[0].replace(": ", ":<br>") for label in labels])
    figure.update_layout(
        title=("Test-case comparison — diamonds show perfect information"
               + (", crosses the no-storage baseline" if has_baseline else "")),
        template="plotly_white",
        barmode="group",
        height=480, yaxis_title=CASE_COMPARISON_METRICS[metric],
        margin={"l": 70, "r": 25, "t": 75, "b": 85},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01,
                "xanchor": "left", "x": 0},
    )
    return figure


def build_cost_breakdown_figures(job, category="imperfect", selected_cases=None,
                                 selected_storage=None, selected_strategies=None):
    """Plot disjoint cost components, one stack per completed case and cost basis."""
    if job.get("information_mode") == "perfect_only":
        category = "perfect"
    rows = []
    for key, result in job.get("results", {}).items():
        short, storage = key.split("|")[:2]
        if ((selected_cases is not None and key not in selected_cases)
                or (selected_storage is not None and storage not in selected_storage)
                or (selected_strategies is not None and short not in selected_strategies)):
            continue
        if category == "imperfect_with_faults" and getattr(result, "imperfect_with_faults", None) is None:
            continue
        _, economics, information_label = _result_category_parts(result, category)
        if economics is not None:
            rows.append((key, economics, information_label))
    if not rows:
        message = "No completed cases are available for the selected filters and result category."
        return empty_comparison_figure(message), empty_comparison_figure(message)
    labels = [job.get("case_labels", {}).get(key, key) for key, _, _ in rows]
    ticks = [label.split(" \u00b7 ")[0].replace(": ", ":<br>") for label in labels]
    palette = ["#d2a449", "#276b72", "#549e9a", "#6c80ad", "#a582b5", "#ad704d",
               "#72a16c", "#c97681", "#879aa8", "#4b789f", "#7c9757", "#b6a36a"]
    figures = []
    for field, title, units, total_field in (
        ("capex_breakdown_usd", "Capital cost", "Illustrative USD", "total_capex_usd"),
        ("opex_breakdown_usd_per_year", "Annual operating cost", "Illustrative USD/year", "annual_opex_usd_per_year"),
    ):
        if any(field not in economics for _, economics, _ in rows):
            figures.append(empty_comparison_figure("Run the cases again to calculate the cost breakdown."))
            continue
        components = list(dict.fromkeys(component for _, economics, _ in rows
                                        for component in economics[field]))
        figure = go.Figure()
        for index, component in enumerate(components):
            values = [economics[field].get(component, 0.0) for _, economics, _ in rows]
            if not any(values):
                continue
            figure.add_bar(
                name=component, x=labels, y=values,
                marker_color=palette[index % len(palette)], marker_line_width=0.5,
                marker_line_color="white",
                customdata=[[economics[total_field]] for _, economics, _ in rows],
                hovertemplate=("%{x}<br>" + component + ": %{y:,.0f} " + units
                               + "<br>Case total: %{customdata[0]:,.0f}<extra></extra>"),
            )
        figure.update_layout(
            title=f"{title} - {rows[0][2]}", template="plotly_white", barmode="stack",
            height=540, yaxis_title=units, bargap=0.35,
            margin={"l": 70, "r": 25, "t": 65, "b": 140},
            legend={"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0},
        )
        figure.update_xaxes(tickmode="array", tickvals=labels, ticktext=ticks, automargin=True)
        figure.update_yaxes(rangemode="tozero")
        figures.append(figure)
    return tuple(figures)


def _update_job(job_id: str, message: str | None = None, **values) -> None:
    with RUN_JOBS_LOCK:
        job = RUN_JOBS[job_id]
        if message and (not job["messages"] or job["messages"][-1] != message):
            job["messages"].append(message)
        job.update(values)


def _load_demo_bundle(paths) -> tuple[list[str], int, dict | None]:
    """Restore whichever shipped result files are present.

    Returns what was loaded, how many cached weather years came with it, and the
    job holding any single-site cases.
    """
    loaded, restored, job = [], 0, None
    for path in paths:
        if not path.exists():
            continue
        kind, payload, weather = import_results(path.read_bytes())
        restored += restore_weather_cache(weather)
        if kind == "showcase":
            with RUN_JOBS_LOCK:
                SHOWCASE_RESULTS.update({row["site"]: row for row in payload})
            loaded.append(f"{len(payload)} showcase location"
                          + ("" if len(payload) == 1 else "s"))
        else:
            job = {"job_id": _register_loaded_results(payload)}
            count = len(payload["results"])
            loaded.append(f"{count} single-site case" + ("" if count == 1 else "s"))
    return loaded, restored, job


def _register_loaded_results(payload: dict) -> str:
    """Register an imported single-site payload as an already completed job."""
    job_id = uuid4().hex
    with RUN_JOBS_LOCK:
        RUN_JOBS[job_id] = {
            **{key: payload[key] for key in JOB_FIELDS if key in payload},
            "status": "complete",
            "messages": ["Results loaded from file; no simulation was run."],
            "delivered_selection": None, "delivered_failure": False,
            "delivered_comparison": None,
        }
        completed = [key for key, value in RUN_JOBS.items()
                     if key != job_id and not value.get("internal")
                     and value.get("status") in {"complete", "failed"}]
        for old_job_id in completed[:-MAX_COMPLETED_JOBS]:
            RUN_JOBS.pop(old_job_id, None)
    return job_id


def _job_snapshot(job_id: str) -> dict | None:
    with RUN_JOBS_LOCK:
        job = RUN_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _result_key(short_strategy: str, long_storage: str) -> str:
    return f"{short_strategy}|{long_storage}"


def _compact_dashboard_result(result, *, perfect_only: bool, plant=None,
                              ambient_temperature=None) -> tuple[float, float]:
    """Retain graph columns for every selectable category and release the rest."""
    simulations = [
        result.perfect, result.imperfect,
        getattr(result, "imperfect_with_faults", None),
        # The no-storage baseline is a selectable result category like any other, so
        # it was being carried at full width — all sixty-odd raw columns plus the
        # per-hour state label — while the three it is compared against were trimmed.
        getattr(result, "baseline", None),
    ]
    before_bytes = sum(
        int(simulation.hourly.memory_usage(index=True, deep=True).sum())
        for simulation in simulations if simulation is not None
    )
    if perfect_only and result.perfect is None:
        raise ValueError("Selected dashboard simulation is missing")
    for simulation in simulations:
        if simulation is None:
            continue
        if plant is not None:
            simulation.metadata["pfd_summary"] = summarize_streams(
                simulation, plant, result.energy, ambient_temperature
            )
        columns = [column for column in DASHBOARD_HOURLY_COLUMNS
                   if column in simulation.hourly.columns]
        simulation.hourly = simulation.hourly.loc[:, columns].copy()
        simulation.daily = pd.DataFrame()
    after_bytes = sum(
        int(simulation.hourly.memory_usage(index=True, deep=True).sum())
        for simulation in simulations if simulation is not None
    )
    gc.collect()
    return before_bytes / 1_000_000, after_bytes / 1_000_000


def compact_for_demo(payload):
    """Cut a completed single-site payload down to what a demo bundle needs to plot.

    A shipped demo is read, never re-analysed, so it keeps only the columns the
    dashboard reads back — see DISPLAY_HOURLY_COLUMNS. Metrics, economics, sizing and
    the PFD summary are untouched: they are stored conclusions, not hourly traces.
    """
    payload = dict(payload)
    for result in (payload.get("results") or {}).values():
        for simulation in (result.perfect, result.imperfect,
                           getattr(result, "imperfect_with_faults", None),
                           getattr(result, "baseline", None)):
            if simulation is None or simulation.hourly.empty:
                continue
            columns = [column for column in DISPLAY_HOURLY_COLUMNS
                       if column in simulation.hourly.columns]
            simulation.hourly = simulation.hourly.loc[:, columns].copy()
            simulation.daily = pd.DataFrame()
    gc.collect()
    return payload


def _result_category_parts(result, category: str):
    if category == "perfect":
        return result.perfect, result.economics_perfect, "perfect information"
    if category == "baseline" and getattr(result, "baseline", None) is not None:
        return result.baseline, result.economics_baseline, "no-storage baseline"
    if category == "imperfect_with_faults" and getattr(result, "imperfect_with_faults", None) is not None:
        return (
            result.imperfect_with_faults,
            result.economics_imperfect_with_faults,
            "imperfect with faults",
        )
    return result.imperfect, result.economics_imperfect, "imperfect forecast"


def _sizing_for_category(result, category: str | None):
    """The plant the selected category actually ran.

    Once the forecast-driven cases are sized on the training record they are a
    different plant from the perfect-information one, and reporting the perfect
    plant's capacities beside an imperfect result would describe equipment that
    result never had.
    """
    built = getattr(result, "sizing_imperfect", None)
    if built is not None and category not in (None, "perfect"):
        return built
    return result.sizing


def _equipment_register_for_category(result, category: str | None):
    built = getattr(result, "equipment_sizing_imperfect", ())
    if built and category not in (None, "perfect"):
        return built
    return result.equipment_sizing


def _result_cards(result, evaluation_period: str | None = None,
                  information_mode: str = "comparison", *,
                  result_category: str | None = None,
                  short_strategy: str | None = None,
                  long_storage: str | None = None) -> list:
    perfect_only = information_mode == "perfect_only"
    result_category = "perfect" if perfect_only else (result_category or "imperfect")
    simulation, economics, category_label = _result_category_parts(result, result_category)
    if simulation is None or economics is None:
        return []
    sizing = _sizing_for_category(result, result_category)
    metrics = simulation.metrics
    short_installed = simulation.metadata["short_storage"]["capacity_kwh"]
    long_installed = simulation.metadata["long_storage"]["capacity_kwh"]
    methane_storage = simulation.metadata["methane_storage"]
    power = simulation.metadata["storage_power"]

    def lcom_text(economics_case: dict) -> str:
        value = economics_case.get("lcom_usd_per_kg_ch4")
        return f"${value:,.2f}/kg" if value is not None else "undefined"

    def total_methane_text(simulation_case) -> str:
        return f"{simulation_case.metrics['methane_total_kg']:,.0f} kg"

    cards = [
        _number_card("Result category", category_label.title()),
        _number_card("Annual methane", f"{metrics['average_annual_methane_kg']:,.0f} kg/y"),
        _number_card("Perfect-information LCOM", lcom_text(result.economics_perfect)),
        _number_card("Perfect total CH4", total_methane_text(result.perfect)),
        _number_card("Annual cyclic energy deficit",
                     f"{sizing.average_annual_balance_deficit_kwh / 1000:,.2f} MWh/y"),
        _number_card("Annual methane shortfall",
                     f"{metrics['average_annual_methane_shortfall_kg']:,.0f} kg/y"),
        _number_card("Selected-case LCOM", lcom_text(economics)),
        _number_card(
            # Without a vessel the rate is a mean, not a promise.
            "Continuous methane delivery" if methane_storage["capacity_kg"]
            else "Mean methane delivery (floating)",
            f"{methane_storage['continuous_delivery_kg_h']:,.2f} kg/h",
        ),
        _number_card(
            "Product methane buffer vessel",
            f"{methane_storage['capacity_kg']:,.0f} kg CH4",
        ),
        _number_card(
            "Methane buffer CAPEX",
            f"${economics['methane_storage_capex_usd']:,.0f}",
        ),
        _number_card("Utilisation", f"{metrics['plant_utilisation']:.1%}"),
        _number_card("Curtailment", f"{metrics['curtailment_fraction']:.1%}"),
        _number_card(
            "Initial SOC",
            f"{simulation.metadata['long_storage']['initial_soc_fraction']:.0%}",
        ),
        _number_card("Short energy: installed / required",
                     f"{short_installed / 1000:,.1f} / "
                     f"{sizing.short_capacity_kwh / 1000:,.1f} MWh"),
        _number_card("Long energy: installed / required",
                     f"{long_installed / 1000:,.1f} / "
                     f"{sizing.long_capacity_kwh / 1000:,.1f} MWh"),
        _number_card("Short power: installed / required",
                     f"+{power['short_installed_charge_kw'] / 1000:,.2f}/-"
                     f"{power['short_installed_discharge_kw'] / 1000:,.2f} vs "
                     f"+{power['short_required_charge_kw'] / 1000:,.2f}/-"
                     f"{power['short_required_discharge_kw'] / 1000:,.2f} MW"),
        _number_card("Long power: installed / required",
                     f"+{power['long_installed_charge_kw'] / 1000:,.2f}/-"
                     f"{power['long_installed_discharge_kw'] / 1000:,.2f} vs "
                     f"+{power['long_required_charge_kw'] / 1000:,.2f}/-"
                     f"{power['long_required_discharge_kw'] / 1000:,.2f} MW"),
    ]
    if result.imperfect is not None and result.economics_imperfect is not None:
        cards.insert(3, _number_card("Imperfect-information LCOM",
                                     lcom_text(result.economics_imperfect)))
        cards.insert(4, _number_card("Imperfect total CH4",
                                     total_methane_text(result.imperfect)))
    if result_category == "imperfect":
        ratio = economics.get("relative_prediction_cost_ratio")
        cards.append(_number_card("Forecast cost ratio", f"{ratio:.3f}" if ratio else "undefined"))
    elif result_category == "imperfect_with_faults":
        fault_ratio = economics.get("relative_fault_cost_ratio")
        combined_ratio = economics.get("relative_combined_cost_ratio")
        production_ratio = metrics.get("relative_fault_production_ratio")
        cards.extend([
            _number_card("Fault cost ratio", f"{fault_ratio:.3f}" if fault_ratio else "undefined"),
            _number_card("Combined cost ratio", f"{combined_ratio:.3f}" if combined_ratio else "undefined"),
            _number_card("Fault production ratio",
                         f"{production_ratio:.3f}" if production_ratio is not None else "undefined"),
            _number_card("Incremental fault loss",
                         f"{metrics.get('incremental_fault_methane_loss_kg', 0.0):,.0f} kg"),
        ])
    if long_storage:
        cards.insert(0, _number_card("Long storage", LONG_STORAGE_LABELS[long_storage]))
    if long_storage == "h2_co2":
        co2_installed = simulation.metadata["long_storage"].get("co2_capacity_kg", 0.0)
        cards.append(_number_card(
            "CO2 vessel: installed / required",
            f"{co2_installed:,.0f} / {sizing.long_co2_capacity_kg:,.0f} kg",
        ))
        cards.append(_number_card(
            "Unmatched CO2 inventory",
            f"{metrics.get('unmatched_co2_inventory_hours', 0):,.0f} h; "
            f"max {metrics.get('max_unmatched_co2_kg', 0.0):,.0f} kg",
        ))
    if short_strategy:
        cards.insert(0, _number_card("Operating strategy", short_strategy.replace("_", " ").title()))
    if evaluation_period:
        cards.insert(0, _number_card("Evaluation period", evaluation_period))
    return cards


def _execute_single_site_job(job_id: str, parameters: dict, progress_callback=None) -> None:
    def progress(message: str) -> None:
        _update_job(job_id, message)
        if progress_callback:
            progress_callback(message)

    try:
        _update_job(job_id, "Preparing the single-site run.", status="running")
        if (parameters.get("capacity_setting_mode") == "auto"
                and parameters.get("run_scope") in {"all", "investigate"}):
            raise ValueError(
                "Auto-calibrate capacities is available for Run selected case only."
            )
        lat, lon = float(parameters["lat"]), float(parameters["lon"])
        assumptions = default_assumption_config()
        assumptions.update(parameters.get("assumptions") or {})
        plant_values = _assumption_section(
            assumptions, "plant", PlantParameters(), PLANT_ASSUMPTION_META
        )
        plant_values.update(_assumption_section(
            assumptions, "plant", PlantParameters(), PHYSICAL_PROPERTY_ASSUMPTION_META
        ))
        thermal_values = _assumption_section(
            assumptions, "thermal", ThermalParameters(), THERMAL_ASSUMPTION_META
        )
        strategy_values = _assumption_section(
            assumptions, "strategy", StrategyConfig(), STRATEGY_ASSUMPTION_META
        )
        weather_values = _assumption_section(
            assumptions, "weather", WeatherConfig(lat, lon), WEATHER_ASSUMPTION_META
        )
        synthetic_values = _assumption_section(
            assumptions, "synthetic", SyntheticWeatherParameters(),
            SYNTHETIC_WEATHER_ASSUMPTION_META,
        )
        economic = EconomicParameters(costs=_economic_costs_from_assumptions(assumptions))
        compare_information = parameters["information_mode"] == "comparison"
        api_training_years = weather_values["training_years"] if compare_information else 0
        api_evaluation_years = weather_values["evaluation_years"]
        if parameters["weather_source"] in {"ninja", "cached"}:
            # Cached-only mode never issues a request, so exploring settings against
            # already downloaded years cannot spend metered calls by accident.
            allow_api = parameters["weather_source"] == "ninja"
            api_token = _resolve_ninja_api_key(parameters.get("ninja_api_key")) if allow_api else None
            if allow_api and not api_token:
                raise WeatherDataError(
                    "Renewables.ninja requires an API key. Enter one in the app "
                    "or set RENEWABLES_NINJA_TOKEN in .env."
                )
            weather, source_metadata = fetch_solar_profile(
                WeatherConfig(
                    lat, lon, latest_year=int(parameters["latest_year"]),
                    training_years=api_training_years,
                    evaluation_years=api_evaluation_years,
                    **{key: value for key, value in weather_values.items()
                       if key not in {"training_years", "evaluation_years"}},
                ),
                token=api_token,
                allow_api=allow_api,
                progress=progress,
            )
            training, actual = split_weather_period(
                weather, training_years=api_training_years,
                evaluation_years=api_evaluation_years,
            )
        else:
            progress("Weather: generating the deterministic offline profile.")
            synthetic_start_year = int(assumptions["synthetic.start_year"])
            synthetic_training_years = int(assumptions["synthetic.training_years"])
            synthetic_evaluation_years = int(assumptions["synthetic.evaluation_years"])
            synthetic_seed = int(abs(
                lat * float(assumptions["synthetic.seed_latitude_multiplier"])
                + lon * float(assumptions["synthetic.seed_longitude_multiplier"])
            ))
            weather = make_synthetic_weather(
                f"{synthetic_start_year}-01-01",
                years=synthetic_training_years + synthetic_evaluation_years,
                latitude_deg=lat, seed=synthetic_seed,
                parameters=SyntheticWeatherParameters(**synthetic_values),
            )
            training, actual = split_weather_period(
                weather, training_years=synthetic_training_years,
                evaluation_years=synthetic_evaluation_years,
            )
            source_metadata = {
                "source": "synthetic_demo",
                "training_years": sorted(int(year) for year in training.index.year.unique()),
                "evaluation_years": sorted(int(year) for year in actual.index.year.unique()),
            }
        if compare_information:
            progress(f"Forecast: building the {len(training.index.year.unique())}-year hourly climatology.")
            forecast = build_climatology_forecast(training, actual)
        else:
            progress("Forecast: perfect-information-only mode selected; climatology skipped.")
            forecast = None
        evaluation_years = sorted(int(year) for year in actual.index.year.unique())
        evaluation_period = (str(evaluation_years[0]) if len(evaluation_years) == 1
                             else f"{evaluation_years[0]}–{evaluation_years[-1]}")
        # Scenario controls on the Single site tab override the saved assumption
        # defaults. Fall back to the dataclass default when a caller omits one.
        plant = replace(
            PlantParameters(**plant_values),
            solar_farm_mw=float(parameters["farm_mw"]),
            air_exhaust_hx_effectiveness=float(
                parameters.get("air_hx_effectiveness")
                if parameters.get("air_hx_effectiveness") is not None
                else PlantParameters().air_exhaust_hx_effectiveness
            ),
        )
        thermal = ThermalParameters(**thermal_values)
        initial_soc_fraction = float(parameters.get("initial_soc_fraction", 0.50))
        short_storage_values = {
            name: float(assumptions[f"storage.short_battery.{name}"])
            for name in ("self_discharge_fraction_per_h",)
        }
        short_store = StorageParameters(
            method="battery", **short_storage_values,
            initial_soc_fraction=initial_soc_fraction,
        )
        faults_enabled = bool(parameters.get("faults_enabled", False)) and compare_information
        fault_scenario = None
        faults: tuple[FaultEvent, ...] = ()
        if faults_enabled:
            fault_config = parameters.get("fault_config") or {}

            def distribution(component: str) -> FaultDistribution:
                values = fault_config.get(component, {})
                return FaultDistribution(
                    mean_months_between_faults=values.get("months", 6),
                    mean_duration_h=values.get("duration_h", 48),
                    mean_capacity_fraction=values.get("capacity_fraction", 0.80),
                )

            fault_scenario = FaultScenario(
                battery=distribution("battery"),
                hydrogen=distribution("hydrogen"),
                sabatier=distribution("sabatier"),
                dac=distribution("dac"),
                seed=parameters.get("fault_seed", 0),
            )
            faults = generate_fault_events(actual.index, fault_scenario)
            counts = {
                component: sum(event.component == component for event in faults)
                for component in ("battery", "hydrogen", "sabatier", "dac")
            }
            progress(
                "Fault schedule: generated "
                + ", ".join(f"{count} {component}" for component, count in counts.items())
                + f" events from seed {fault_scenario.seed}."
            )
        if parameters.get("run_scope") == "all":
            combinations = [
                (short_name, long_method)
                for short_name in OPERATING_STRATEGIES
                for long_method in LONG_STORAGE_METHODS
            ]
        else:
            combinations = [(parameters["short_name"], parameters["long_method"])]
        if parameters.get("run_scope") == "investigate":
            cases = investigation_cases(parameters)
        else:
            cases = [dict(parameters, short_name=short, long_method=long, label="")
                     for short, long in combinations]
        results = {}
        case_labels = {}
        calibration_summary = None
        for number, case in enumerate(cases, start=1):
            short_name, long_method = case["short_name"], case["long_method"]
            case_key = _result_key(short_name, long_method)
            if case_key in results:
                case_key += f"|{case['reactor_count']}|{case['reactor_scheduling_mode']}"
            case_labels[case_key] = (
                f"{case['label'] or short_name.replace('_', ' ').title()} · "
                f"{LONG_STORAGE_LABELS[long_method]} · {case.get('reactor_count', 1)} trains · "
                f"{case.get('reactor_scheduling_mode', 'seasonal').replace('_', ' ')}"
            )
            label = f"Case {number}/{len(cases)} ({case_labels[case_key]})"
            progress(f"{label}: starting calculations.")
            strategy = StrategyConfig(
                short_strategy=short_name,
                long_strategy=strategy_values["long_strategy"],
                f_ocp=float(parameters["f_ocp"]),
                f_socp_long=float(parameters["f_socp"]),
                daylight_cf_cutoff=strategy_values["daylight_cf_cutoff"],
                parallel_reactor_count=int(case.get("reactor_count", 1)),
                reactor_scheduling_mode=case.get(
                    "reactor_scheduling_mode", strategy_values["reactor_scheduling_mode"]
                ),
                storage_planning_lookahead_days=strategy_values["storage_planning_lookahead_days"],
                # The product vessel is this method's seasonal store, not an extra
                # buffer bolted onto one. Every other method keeps the plant running
                # through winter instead, so methane leaves the gate as it is made.
                product_storage=long_method == PRODUCT_STORAGE_METHOD,
            )
            storage_prefix = {
                "battery": "storage.long_battery",
                "hydrogen": "storage.long_hydrogen",
                "h2_co2": "storage.long_h2_co2",
                # No upstream store is built, so its loss rate never applies; the
                # battery section is read only to keep the lookup total.
                "methane": "storage.long_battery",
            }[long_method]
            long_storage_values = {
                name: float(assumptions[f"{storage_prefix}.{name}"])
                for name in (
                    ("self_discharge_fraction_per_h", "co2_self_discharge_fraction_per_h")
                    if long_method == "h2_co2" else ("self_discharge_fraction_per_h",)
                )
            }
            long_store = StorageParameters(
                method="battery" if long_method == PRODUCT_STORAGE_METHOD else long_method,
                **long_storage_values,
                initial_soc_fraction=initial_soc_fraction,
            )
            if parameters.get("capacity_setting_mode") == "auto":
                progress(
                    f"{label}: starting perfect-information capacity calibration "
                    f"with a maximum of {MAX_CAPACITY_CALIBRATION_EVALUATIONS} evaluations."
                )
                calibration = calibrate_capacity_factors(
                    actual, forecast, plant=plant, thermal=thermal, strategy=strategy,
                    short_storage_template=short_store,
                    long_storage_template=long_store, economic=economic,
                    factor_minimum=float(parameters.get("factor_minimum", 0.0)),
                    factor_maximum=float(parameters.get("factor_maximum", 2.0)),
                    factor_increment=float(parameters.get("factor_increment", 0.05)),
                    max_evaluations=MAX_CAPACITY_CALIBRATION_EVALUATIONS,
                    progress=lambda message, prefix=label: progress(f"{prefix}: {message}"),
                )
                strategy = calibration.strategy
                calibration_summary = {
                    "f_ocp": calibration.optimized_factors[0],
                    "f_socp": calibration.optimized_factors[1],
                    "starting_factors": calibration.starting_factors,
                    "starting_lcom_usd_per_kg_ch4": calibration.starting_lcom_usd_per_kg_ch4,
                    "optimized_lcom_usd_per_kg_ch4": calibration.optimized_lcom_usd_per_kg_ch4,
                    "evaluations": calibration.evaluations,
                    "converged": calibration.converged,
                    "limit_reached": calibration.limit_reached,
                    "feasible": calibration.feasible,
                    "average_annual_balance_deficit_kwh": calibration.average_annual_balance_deficit_kwh,
                }
            case_metadata = dict(source_metadata)
            case_metadata["matrix_case"] = {
                "short_strategy": short_name, "long_storage": long_method,
                "parallel_reactor_count": strategy.parallel_reactor_count,
                "reactor_scheduling_mode": strategy.reactor_scheduling_mode,
            }
            result = run_case(
                actual, forecast, plant=plant, thermal=thermal, strategy=strategy,
                short_storage_template=short_store, long_storage_template=long_store,
                economic=economic, faults=faults, fault_scenario=fault_scenario,
                include_faulted=faults_enabled, metadata=case_metadata,
                include_imperfect=compare_information,
                include_baseline=bool(parameters.get("include_baseline", True)),
                # The forecast-driven cases run a plant sized on the training record,
                # the way a designer building in the first evaluation year would have
                # had to. Perfect-only runs have no training block to size from, so
                # they keep sizing on the evaluation period.
                sizing_profile=training if compare_information and not training.empty
                               else None,
                progress=lambda message, prefix=label: progress(f"{prefix}: {message}"),
            )
            if calibration_summary is not None:
                result.perfect.metadata["capacity_calibration"] = calibration_summary
                if result.imperfect is not None:
                    result.imperfect.metadata["capacity_calibration"] = calibration_summary
                if getattr(result, "imperfect_with_faults", None) is not None:
                    result.imperfect_with_faults.metadata["capacity_calibration"] = calibration_summary
                if calibration_summary["limit_reached"]:
                    warning = (
                        f"Capacity calibration reached the {MAX_CAPACITY_CALIBRATION_EVALUATIONS}-evaluation limit; "
                        "the best candidate found was used."
                    )
                    result.perfect.warnings.append(warning)
                    if result.imperfect is not None:
                        result.imperfect.warnings.append(warning)
                    if getattr(result, "imperfect_with_faults", None) is not None:
                        result.imperfect_with_faults.warnings.append(warning)
                if not calibration_summary["feasible"]:
                    warning = (
                        "Capacity calibration found no cyclically feasible candidate; "
                        "the lowest-deficit candidate found was used."
                    )
                    result.perfect.warnings.append(warning)
                    if result.imperfect is not None:
                        result.imperfect.warnings.append(warning)
                    if getattr(result, "imperfect_with_faults", None) is not None:
                        result.imperfect_with_faults.warnings.append(warning)
            before_mb, after_mb = _compact_dashboard_result(
                result, perfect_only=not compare_information, plant=plant,
                ambient_temperature=actual.get("ambient_temperature_k")
            )
            results[case_key] = result
            progress(
                f"{label}: complete; retained {after_mb:,.1f} MB of graph data "
                f"from {before_mb:,.1f} MB of hourly results."
            )
        finished_message = f"{len(cases)} test cases complete." if len(cases) > 1 else "Selected case complete."
        finished_message += (
            " Storage transfer remained unconstrained; perfect-information peak "
            "flows set the costed equipment ratings."
        )
        _update_job(job_id, finished_message, status="complete",
                    results=results, case_labels=case_labels, source_metadata=source_metadata,
                    evaluation_period=evaluation_period,
                    information_mode=parameters["information_mode"],
                    inputs=safe_inputs(parameters),
                    assumptions=assumptions,
                    faults_enabled=faults_enabled,
                    fault_scenario=(asdict(fault_scenario) if fault_scenario is not None else None),
                    optimized_factors=calibration_summary,
                    run_scope=parameters.get("run_scope", "specific"),
                    delivered_selection=None, delivered_failure=False)
    except Exception as exc:
        _update_job(job_id, f"Run failed: {exc}", status="failed", error=str(exc),
                    delivered_failure=False)


def _execute_showcase_job(job_id, locations, parameters):
    """Run one base case per location, retaining only map summaries between sites."""
    rows, errors = [], []
    for number, location in enumerate(locations, 1):
        _update_job(job_id, f"Location {number}/{len(locations)}: {location['site']}",
                    status="running")
        child_id = uuid4().hex
        # Showcase runs always compute all three information/fault variants so no map
        # metric can be blank; the status label has to report what actually ran.
        run_parameters = {**parameters, **location, "run_scope": "specific",
                          "information_mode": "comparison", "faults_enabled": True,
                          "include_baseline": True}
        with RUN_JOBS_LOCK:
            RUN_JOBS[child_id] = {"status": "queued", "messages": [], "internal": True}
        try:
            _execute_single_site_job(
                child_id, run_parameters,
                progress_callback=lambda message: _update_job(
                    job_id, f"Location {number}/{len(locations)} ({location['site']}): {message}"),
            )
            child = _job_snapshot(child_id)
            if child["status"] != "complete":
                raise ValueError(child.get("error", "Case did not complete"))
            result = next(iter(child["results"].values()))
            simulation = result.imperfect if result.imperfect is not None else result.perfect
            economics = (result.economics_imperfect if result.imperfect is not None
                         else result.economics_perfect)
            metrics = simulation.metrics
            rows.append({
                **location,
                "data_status": f"{child['source_metadata'].get('source', 'model')} / "
                               f"{child['evaluation_period']} / "
                               f"{run_parameters['short_name']} / "
                               f"{run_parameters['long_method']} / "
                               f"{run_parameters['information_mode']}",
                **{key: metrics[key] for key in (
                    "average_annual_methane_kg", "plant_utilisation", "curtailment_fraction")},
                **{key: economics[key] for key in (
                    "lcom_usd_per_kg_ch4", "total_capex_usd", "annual_opex_usd_per_year")},
                "annual_balance_deficit_mwh": result.sizing.average_annual_balance_deficit_kwh / 1000,
                # The plant sized on the training record can be under-built for the
                # decade that follows, which shows up as a cheap LCOM unless the map
                # says why. Recorded per site so the hover can flag it.
                "imperfect_deficit_mwh": (
                    getattr(result, "imperfect_realised_deficit_kwh_per_year", 0.0) / 1000),
                "short_storage_capacity_mwh":
                    simulation.metadata["short_storage"]["capacity_kwh"] / 1000,
                "long_storage_capacity_mwh":
                    simulation.metadata["long_storage"]["capacity_kwh"] / 1000,
                # Still written because the saved-file format requires it.
                "storage_capacity_mwh": (simulation.metadata["short_storage"]["capacity_kwh"]
                    + simulation.metadata["long_storage"]["capacity_kwh"]) / 1000,
                "forecast_cost_ratio": economics.get("relative_prediction_cost_ratio"),
                "inputs": {**safe_inputs(parameters), **location,
                           "assumptions": child["assumptions"],
                           "optimized_factors": child.get("optimized_factors")},
                "case_id": result.case_id,
                "source_metadata": child["source_metadata"],
                "outputs": {
                    "economics_perfect": result.economics_perfect,
                    "economics_imperfect": result.economics_imperfect,
                    "economics_imperfect_with_faults": result.economics_imperfect_with_faults,
                    "economics_baseline": result.economics_baseline,
                    "perfect_metrics": result.perfect.metrics,
                    "imperfect_metrics": result.imperfect.metrics if result.imperfect is not None else None,
                    "equipment_sizing": result.equipment_sizing,
                },
            })
            # JSON nulls keep unavailable metrics safe for browser stores.
            rows[-1] = normalize_showcase_ratio(rows[-1])
            rows = json.loads(pd.DataFrame(rows).to_json(orient="records"))
            with RUN_JOBS_LOCK:
                SHOWCASE_RESULTS[location["site"]] = dict(rows[-1])
        except Exception as exc:
            errors.append(f"{location['site']}: {exc}")
        finally:
            with RUN_JOBS_LOCK:
                RUN_JOBS.pop(child_id, None)
        _update_job(job_id, rows=list(rows), errors=list(errors), finished=number)
    _update_job(job_id, f"{len(rows)} of {len(locations)} locations completed.",
                status="complete", rows=rows, errors=errors)


def build_location_picker(selected_lat: float = 51.5074, selected_lon: float = -0.1278):
    """Dense clickable coordinate targets plus the currently selected marker."""
    latitudes, longitudes, coordinates = [], [], []
    for lat_half_degree in range(70, 143):
        lat = lat_half_degree / 2
        for lon_half_degree in range(-30, 73):
            lon = lon_half_degree / 2
            latitudes.append(lat)
            longitudes.append(lon)
            coordinates.append([lat, lon])
    figure = go.Figure()
    figure.add_trace(go.Scattermap(lat=latitudes, lon=longitudes, mode="markers",
                                   customdata=coordinates,
                                   marker={"size": MAP_GRID_SIZE, "opacity": 0.08,
                                           "color": "#176b55"},
                                   hovertemplate="%{lat:.2f}, %{lon:.2f}<extra>select</extra>",
                                   name="Clickable locations"))
    figure.add_trace(go.Scattermap(
        lat=[selected_lat], lon=[selected_lon], customdata=[[selected_lat, selected_lon]],
        mode="markers", marker={"size": 16, "color": "#cf3c3c"}, name="Selected",
        hovertemplate="Selected: %{lat:.4f}, %{lon:.4f}<extra></extra>",
    ))
    figure.update_layout(map={"style": "open-street-map", "center": {"lat": 52, "lon": 10}, "zoom": 2.2},
                         height=360, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                         uirevision="location-picker",
                         title="Click a faint green point or enter exact coordinates")
    return figure


def build_sizing_table(result, simulation=None, category=None) -> html.Div:
    """Render the equipment sizing register for the plant this category ran."""
    headings = (
        "Category", "Sized unit", "Count", "Required per unit",
        "Required total", "Installed per unit", "Installed total", "Unit", "Sizing basis",
    )
    keys = (
        "category", "unit_name", "count", "required_size_per_unit",
        "required_total_size", "installed_size_per_unit", "installed_total_size",
        "unit", "sizing_basis",
    )

    def display(key, value):
        if key in {"required_size_per_unit", "required_total_size",
                   "installed_size_per_unit", "installed_total_size"}:
            return f"{float(value):,.3g}"
        return value

    header = html.Tr([
        html.Th(label, style={"padding": "0.65rem", "textAlign": "left",
                              "borderBottom": "2px solid #aac2ba", "background": "#edf5f2"})
        for label in headings
    ])
    sizing_rows = []
    for source_row in _equipment_register_for_category(result, category):
        row = dict(source_row)
        if (simulation is not None
                and row["unit_name"] == "Product methane buffer vessel"):
            capacity = simulation.metadata["methane_storage"]["capacity_kg"]
            row.update({
                "required_size_per_unit": capacity,
                "required_total_size": capacity,
                "installed_size_per_unit": capacity,
                "installed_total_size": capacity,
            })
        sizing_rows.append(row)
    body = [
        html.Tr([
            html.Td(display(key, row[key]), style={
                "padding": "0.55rem 0.65rem", "verticalAlign": "top",
                "borderBottom": "1px solid #e2e8e5",
                "whiteSpace": "normal" if key == "sizing_basis" else "nowrap",
            }) for key in keys
        ], style={"background": "#f7faf9" if index % 2 else "white"})
        for index, row in enumerate(sizing_rows)
    ]
    return html.Div([
        html.H2("Equipment sizing"),
        html.P(
            "Process and input-storage sizes come from the perfect-information case. "
            "The product methane buffer reflects the selected result category. Short-term "
            "storage is installed at exactly its calculated daily-cycle requirement; "
            "long-term storage includes the selected f_SOCP factor. Storage transfer power "
            "is unconstrained during dispatch; the table reports the peak power observed "
            "after the run and used for costing.",
            style={"color": "#40534d"},
        ),
        html.Div(
            html.Table([html.Thead(header), html.Tbody(body)],
                       style={"width": "100%", "borderCollapse": "collapse"}),
            style={"overflowX": "auto", "border": "1px solid #d8e3df", "borderRadius": "8px"},
        ),
    ])


def _fault_distribution_controls(component: str, label: str) -> html.Div:
    marks_months = {0: "Off", 1: "1", 6: "6", 24: "24", 60: "60", 120: "120"}
    marks_duration = {1: "1", 48: "48", 168: "168", 360: "360", 720: "720"}
    marks_capacity = {0: "0%", 0.5: "50%", 0.8: "80%", 1: "100%"}

    def slider(title: str, component_id: str, minimum: float, maximum: float,
               step: float, value: float, marks: dict) -> html.Div:
        return html.Div([
            html.Label(title, style={"fontWeight": 600, "fontSize": "0.86rem"}),
            dcc.Slider(
                id=component_id, min=minimum, max=maximum, step=step, value=value,
                marks=marks, included=False,
                tooltip={"placement": "bottom", "always_visible": False},
                persistence=True, persistence_type="local",
            ),
        ])

    return html.Div([
        html.Strong(label, style={"color": "#173f35"}),
        slider("Mean interval (months)", f"fault-{component}-months",
               0, 120, 1, 6, marks_months),
        slider("Mean duration (h)", f"fault-{component}-duration",
               1, 720, 1, 48, marks_duration),
        slider("Mean retained capacity", f"fault-{component}-capacity",
               0, 1, 0.05, 0.8, marks_capacity),
    ], style={
        "display": "grid", "gridTemplateColumns": "150px repeat(3,minmax(190px,1fr))",
        "gap": "0.8rem", "alignItems": "center", "padding": "0.8rem",
        "borderTop": "1px solid #d8e3df",
    })


def create_app() -> Dash:
    app = Dash(__name__, title="SOLAR BALANCER | Solar-to-Methane Siting & Dispatch")
    input_style = {"width": "100%", "height": "38px", "boxSizing": "border-box"}
    controls = html.Div(
        [
            _control("Latitude", dcc.Input(id="lat", type="number", value=51.5074, step=0.01, style=input_style)),
            _control("Longitude", dcc.Input(id="lon", type="number", value=-0.1278, step=0.01, style=input_style)),
            _control("Solar farm (MW)", dcc.Input(id="farm-mw", type="number", value=10, min=0.1, style=input_style)),
            _control("Weather source", html.Div([
                dcc.RadioItems(
                    id="weather-source",
                    options={"demo": "Offline synthetic", "ninja": "Renewables.ninja",
                             "cached": "Cached only"},
                    value="demo", inline=True, persistence=True, persistence_type="local",
                    style={"minHeight": "38px", "display": "flex",
                           "alignItems": "center", "gap": "0.35rem 0.8rem",
                           "flexWrap": "wrap"},
                ),
                html.Div(id="ninja-usage", style={"fontSize": "0.82rem", "color": "#40534d"}),
                dcc.Interval(id="ninja-usage-poll", interval=5000),
                html.Small(
                    "Renewables.ninja requires an API key; configure it in .env "
                    "or enter it in the adjacent field. Cached only replays weather "
                    "already on disk and never contacts the API.",
                    style={"color": "#60716c"},
                ),
                html.Button("Use cached weather", id="load-cached-weather", n_clicks=0,
                            style={"marginTop": "0.4rem"}),
                dcc.Dropdown(id="cached-site", options=cached_site_options(),
                             value=None, clearable=True,
                             placeholder="Jump to a cached site",
                             style={"marginTop": "0.35rem"}),
                html.Div(id="cached-weather-status", role="status",
                         style={"fontSize": "0.8rem", "color": "#40534d",
                                "marginTop": "0.25rem"}),
            ])),
            _control("Renewables.ninja API key", html.Div([
                dcc.Input(
                    id="ninja-api-key", type="password",
                    placeholder="Enter API key", autoComplete="off", style=input_style,
                ),
                html.Div(
                    id="ninja-api-key-warning",
                    style={"color": "#a21d1d", "fontSize": "0.82rem",
                           "fontWeight": 600, "marginTop": "0.35rem"},
                ),
            ])),
            _control("Information mode", dcc.RadioItems(
                id="information-mode",
                options=[
                    {"label": "Perfect only", "value": "perfect_only"},
                    {"label": "Perfect vs imperfect", "value": "comparison"},
                ],
                value="comparison", inline=True,
                style={"minHeight": "38px", "display": "flex", "alignItems": "center",
                       "gap": "0.8rem", "flexWrap": "wrap"},
            )),
            _control("Latest year (API)", dcc.Input(id="latest-year", type="number", value=2025, min=2000, style=input_style)),
            _control("Parallel Sabatier trains", dcc.Slider(
                id="reactor-count", min=1, max=6, step=1, value=4,
                marks={value: str(value) for value in range(1, 7)},
                included=False,
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type="local",
            )),
            html.Div(
                "Reactor parallelisation is an experimental feature and its sizing and dispatch rules are still being validated.",
                style={"gridColumn": "1 / -1", "color": "#8a4b08", "fontSize": "0.88rem",
                       "marginTop": "-0.55rem"},
            ),
            _control("Reactor scheduling", dcc.RadioItems(
                id="reactor-scheduling-mode",
                options=[
                    {"label": html.Span([
                        "Perfect daily switching ",
                        _experimental(
                            "Experimental: daily switching can behave unpredictably "
                            "when combined with imperfect weather information. Use "
                            "seasonal parallelisation for results you intend to quote."),
                    ]), "value": "daily_storage_aware"},
                    {"label": "Seasonal parallelisation", "value": "seasonal"},
                ],
                value="seasonal", inline=True, persistence=True, persistence_type="local",
                style={"minHeight": "38px", "display": "flex", "alignItems": "center",
                       "gap": "0.8rem", "flexWrap": "wrap"},
            )),
            html.Div(
                id="slow-calibration-warning",
                style={"gridColumn": "1 / -1", "color": "#8a4b08",
                       "fontSize": "0.88rem", "fontWeight": 600},
            ),
            _control("Results: operating strategy", dcc.RadioItems(
                id="short-strategy",
                options=[{"label": value.replace("_", " ").title(), "value": value}
                         for value in OPERATING_STRATEGIES],
                value="limping", inline=True,
                style={"minHeight": "38px", "display": "flex", "alignItems": "center",
                       "gap": "0.8rem", "flexWrap": "wrap"},
            )),
            # Product storage is no longer a switch. It is what "Methane product
            # storage" means as a long-storage method: that method's seasonal store is
            # the vessel, every other method's is upstream of the reactor. The control
            # stays hidden and unpersisted so the callbacks still resolve and a browser
            # that saved a value cannot reinstate a buffer the method does not have.
            html.Div(
                dcc.Checklist(
                    id="product-storage",
                    options=[{"label": " Enable product storage", "value": "enabled"}],
                    value=[],
                ), style={"display": "none"},
            ),
            _control("Results: long storage", dcc.RadioItems(
                id="long-storage",
                options=_long_storage_options(),
                value="battery", inline=True,
                style={"minHeight": "38px", "display": "flex", "alignItems": "center",
                       "gap": "0.8rem", "flexWrap": "wrap"},
            )),
            _control("Initial storage SOC fraction", dcc.Input(
                id="initial-soc", type="number", value=0.50, min=0, max=1,
                step=0.05, style=input_style,
            )),
            _control("Displayed result category", dcc.RadioItems(
                id="result-category",
                options=[
                    {"label": "Perfect information", "value": "perfect"},
                    {"label": "Imperfect forecast", "value": "imperfect"},
                    {"label": "Imperfect with faults", "value": "imperfect_with_faults"},
                ],
                value="imperfect", inline=False,
            )),
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(210px,1fr))",
               "alignItems": "start", "gap": "1rem", "padding": "1rem 0"},
    )
    original_controls = controls.children
    controls = html.Div([
        _section("Site & weather", [
            html.Div(original_controls[:5] + [original_controls[6]],
                     className="control-grid"),
            html.Div(dcc.Graph(id="location-picker", figure=build_location_picker(),
                               config={"responsive": True}),
                     style={"marginTop": "1rem"}),
        ], "#315f8c", True),
        _section("Base case & test categories", html.Div([original_controls[7]] + original_controls[9:15], className="control-grid"), "#176b55", True,
                 "Select the base values below. Include categories to vary them one at a time; all other settings stay at their base values. Alpha: reactor parallelisation is an experimental feature and its sizing and dispatch rules are still being validated."),
        _section("Forecast & result display", html.Div([original_controls[5], original_controls[15]], className="control-grid"), "#7754a3"),
    ])
    fault_controls = html.Div([
        html.Div([
            html.Div([
                dcc.Checklist(
                    id="fault-enabled",
                    options=[{"label": "Enable stochastic faults", "value": "enabled"}],
                    value=[],
                    style={"fontWeight": 700, "color": "#173f35"},
                ),
                html.Span(
                    "The interval N means approximately one event every N months for that subsystem; 0 disables it.",
                    style={"color": "#60716c", "fontSize": "0.86rem"},
                ),
            ], style={"display": "flex", "gap": "1rem", "alignItems": "center",
                      "flexWrap": "wrap"}),
            html.Label([
                "Random seed ",
                dcc.Input(id="fault-seed", type="number", value=0, min=0, step=1,
                          style={"width": "150px", "height": "34px"}),
            ], style={"marginLeft": "auto", "fontWeight": 600}),
        ], style={"display": "flex", "alignItems": "center", "gap": "1rem",
                  "padding": "0.8rem", "flexWrap": "wrap"}),
        html.Div(id="fault-mode-note", role="status",
                 style={"padding": "0 0.8rem 0.6rem", "color": "#8a4b08",
                        "fontSize": "0.85rem", "fontWeight": 600}),
        _fault_distribution_controls("battery", "Battery"),
        _fault_distribution_controls("hydrogen", "Hydrogen apparatus"),
        _fault_distribution_controls("sabatier", "Sabatier reactor"),
        _fault_distribution_controls("dac", "DAC apparatus"),
    ], id="fault-controls", style={
        "border": "1px solid #d8e3df", "borderRadius": "8px",
        "margin": "0.4rem 0 1rem", "overflowX": "auto", "background": "#f8fbfa",
    })
    factor_controls = html.Div([
        html.Div([
            html.H3("Capacity factors", style={"margin": 0, "color": "#173f35"}),
            html.Span("Tune process overcapacity and long-term storage energy before running a case.",
                      style={"color": "#60716c", "fontSize": "0.9rem"}),
        ], style={"gridColumn": "1 / -1", "display": "flex", "gap": "0.7rem",
                  "alignItems": "baseline", "flexWrap": "wrap"}),
        # Auto-calibration is withdrawn from the dashboard but kept in the codebase:
        # calibrate_capacity_factors and its plumbing still work and are still tested,
        # they are simply unreachable from the UI. It searches against
        # perfect-information LCOM, so its answer is tuned to a year the operator could
        # not have known, which is not a default anyone should meet by accident.
        #
        # The control itself stays so every callback still resolves, hidden and pinned
        # to manual. Persistence is deliberately off: a browser that saved "auto"
        # before it was withdrawn would otherwise go on calibrating invisibly, with no
        # visible control to say so.
        html.Div([
            dcc.RadioItems(
                id="capacity-setting-mode",
                options=[{"label": "Set capacities manually", "value": "manual"}],
                value="manual", inline=True,
                style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
            ),
        ], style={"display": "none"}),
        html.Div([
            html.Strong("f_OCP / f_SOCP slider scale", style={"color": "#173f35"}),
            html.Label(["Minimum", dcc.Input(
                id="factor-range-min", type="number", value=0, min=0, step=0.25,
                persistence=True, persistence_type="local",
                style={"width": "84px", "height": "32px", "marginLeft": "0.4rem"},
            )]),
            html.Label(["Maximum", dcc.Input(
                id="factor-range-max", type="number", value=2, min=0.25, step=0.25,
                persistence=True, persistence_type="local",
                style={"width": "84px", "height": "32px", "marginLeft": "0.4rem"},
            )]),
            html.Label(["Increment", dcc.Input(
                id="factor-range-step", type="number", value=0.05, min=0.01, step=0.01,
                persistence=True, persistence_type="local",
                style={"width": "84px", "height": "32px", "marginLeft": "0.4rem"},
            )]),
            html.Span("Scale: 0 to 2 in steps of 0.05, applied to f_OCP and f_SOCP only.",
                      id="factor-range-status",
                      style={"fontSize": "0.82rem", "color": "#60716c"}),
            html.Span("Display only: this rescales the two sliders below and changes "
                      "nothing the model calculates.",
                      id="factor-range-scope",
                      style={"fontSize": "0.82rem", "color": "#60716c",
                             "fontStyle": "italic"}),
        ], style={"gridColumn": "1 / -1", "display": "flex", "gap": "0.9rem",
                  "alignItems": "center", "flexWrap": "wrap", "padding": "0.65rem 0.8rem",
                  "background": "#edf5f2", "borderRadius": "8px"}),
        html.Div([
            html.Strong("Auto-calibration search", style={"color": "#173f35"}),
            html.Span("The space the optimiser explores. Separate from the slider "
                      "scale above, so narrowing the search cannot be done by accident.",
                      style={"fontSize": "0.82rem", "color": "#60716c"}),
            html.Label([
                "Search range for f_OCP and f_SOCP",
                dcc.RangeSlider(
                    id="calibration-range", min=0, max=2, step=0.05, value=[0, 2],
                    marks={0: "0", 0.5: "0.5", 1: "1", 1.5: "1.5", 2: "2"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    persistence=True, persistence_type="local",
                ),
            ], style={"fontWeight": 600, "fontSize": "0.86rem"}),
            html.Label([
                "Search grid",
                dcc.Slider(
                    id="calibration-step", min=0.01, max=0.25, step=0.01, value=0.05,
                    marks={0.01: "0.01", 0.05: "0.05", 0.1: "0.1", 0.25: "0.25"},
                    included=False,
                    tooltip={"placement": "bottom", "always_visible": False},
                    persistence=True, persistence_type="local",
                ),
            ], style={"fontWeight": 600, "fontSize": "0.86rem"}),
        ], id="calibration-controls", style=CALIBRATION_BLOCK_STYLE),
        _factor_slider(
            "Plant sizing reserve · f_OCP", "f-ocp", 0.20,
            "Reduces rated methane capacity relative to the fixed solar farm (0.20 means a 20% reserve denominator).",
        ),
        _factor_slider(
            "Long-term storage energy · f_SOCP", "f-socp", 1.0,
            "Installed long-term energy divided by its calculated requirement; short-term storage is always exactly sized.",
        ),
        _factor_slider(
            "Air/exhaust exchanger effectiveness", "air-hx-effectiveness", 0.97,
            "Fraction of the carbonator air-preheat duty recovered from the depleted "
            "exhaust. Dry calcium looping captures CO2 on hot solids, so the whole air "
            "stream - roughly 2,200 kg of air per kg of CO2 at 400 ppm - has to be "
            "raised to the carbonator temperature. That is inherent to the process, not "
            "a modelling shortcut, and it makes the air by far the largest heat-capacity "
            "flow in the plant. E_req scales with whatever this exchanger does not "
            "recover, which is why it is the most influential single assumption here: "
            "even at 0.97 the residual air duty is still about 39% of E_req, and 0.90 "
            "roughly doubles it.",
            scale=(0.5, 1.0, 0.01),
        ),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(260px,1fr))",
              "gap": "0.85rem", "padding": "0.2rem 0 1.2rem"})
    app.layout = html.Div(
        [
            html.Header([
                html.Div([
                    html.Div("SOLAR / STORAGE / DISPATCH", className="hero-eyebrow"),
                    html.H1("SOLAR BALANCER", className="hero-title"),
                    html.P("A solar-to-methane siting and dispatch tool", className="hero-subtitle"),
                    html.P(
                        "\u00a9 ARPH, 2026 \u2014 Submitted as an entry for the 2026 SOTA "
                        "'Intermittent Abundance' competition",
                        id="hero-attribution", className="hero-attribution",
                    ),
                ], className="hero-intro"),
                html.Div([
                    html.Strong("Disclaimer", className="hero-disclaimer-label"),
                    html.P(
                        "This is a toy model used to demonstrate the parameters influencing "
                        "power-to-X (P2X) chemical production. Calculated values and input "
                        "parameters are indicative only and should not be used for real plant "
                        "design or construction, operational decisions, or investment decisions. "
                        "The model is provided without any guarantee of accuracy, completeness, "
                        "or fitness for a particular purpose. The author bears no responsibility "
                        "for any loss, damage, or consequences arising from its use or reliance "
                        "on its outputs.",
                        id="hero-disclaimer", className="hero-disclaimer-text",
                    ),
                    html.P([
                        "Distributed free of charge under a Creative Commons ",
                        html.A("CC BY-NC 4.0",
                               href="https://creativecommons.org/licenses/by-nc/4.0/",
                               target="_blank", rel="noopener noreferrer",
                               className="hero-licence-link"),
                        " licence — share and adapt with attribution, "
                        "non-commercial use only. See LICENSE.md.",
                    ], className="hero-licence"),
                ], className="hero-disclaimer"),
            ], className="dashboard-hero"),
            demo_panel(),
            dcc.Store(id="run-job"),
            dcc.Store(id="calibrated-factors"),
            dcc.Store(id="saved-assumptions", data=default_assumption_config(),
                      storage_type="local"),
            dcc.Interval(id="run-job-poll", interval=500, n_intervals=0),
            dcc.Tabs([
                dcc.Tab(label="Single site", children=[
                    controls,
                    _section("Fault scenarios", fault_controls, "#b07722"),
                    _section("Capacity factors", factor_controls, "#7754a3"),
                    html.P([
                        "Run selected case calculates the current result-selector combination. ",
                        "Investigations vary one included category at a time, holding other settings at the selected base values. ",
                        "completed metrics without rerunning. Choose a result and load only its plot. ",
                        "Factor definitions: f_OCP reduces rated methane capacity relative to the fixed solar farm; ",
                        "f_SOCP is the installed/required long-term storage-energy ratio ",
                        "(1.00 = exactly sized, 0.75 = 25% undersized). Storage transfer ",
                        "power is unconstrained and the peak required power is reported after each run. Initial SOC is ",
                        "the fraction of installed storage energy present at hour zero and ",
                        "is applied identically to batteries and both inventories in paired H2 + CO2 gas storage.",
                    ], style={"marginTop": "-0.25rem", "color": "#40534d"}),
                    html.Div([
                        html.Button("Run selected case", id="run-specific", n_clicks=0,
                                    style={"fontSize": "1.1rem", "padding": "0.7rem 1.6rem",
                                           "background": "#176b55", "color": "white", "border": 0,
                                           "borderRadius": "7px", "position": "relative", "zIndex": 2}),
                        html.Button("Investigate 0 selected parameters", id="run-all", n_clicks=0, disabled=True,
                                    style={"fontSize": "1.1rem", "padding": "0.7rem 1.6rem",
                                           "background": "#9a5b13", "color": "white", "border": 0,
                                           "borderRadius": "7px", "position": "relative", "zIndex": 2}),
                    ], style={"display": "flex", "gap": "0.8rem", "flexWrap": "wrap"}),
                    html.P("Investigations can take several minutes for a ten-year API profile.",
                           style={"color": "#8a4b08", "fontWeight": 600, "marginTop": "0.6rem"}),
                    html.Div("Ready to run.", id="run-status", style={"marginTop": "1rem"}),
                    _section("Save / load results", [
                        html.Button("Save single-site results", id="save-site-results"),
                        dcc.Upload(id="load-site-results", children=html.Button("Load result file"),
                                   accept=".json,.gz", multiple=False, max_size=MAX_FILE_BYTES),
                        html.P("Saves all completed test cases, including retained hourly/daily data, economics, sizing and PFD summaries. Loading restores results; current input settings are kept."),
                        html.Div("", id="save-size-estimate", role="status",
                                 style={"fontSize": "0.85rem", "fontWeight": 600}),
                        html.Div(id="site-file-status", role="status"),
                        dcc.Download(id="results-download"),
                    ], "#315f8c"),
                    _control("Completed case", dcc.Dropdown(id="completed-case", options=[], value=None, placeholder="Base case", clearable=True)),
                    html.Div(id="metric-cards", style={"display": "flex", "flexWrap": "wrap", "gap": "0.8rem", "margin": "1rem 0"}),
                    html.Button(
                        "Load selected dispatch plot", id="load-dispatch", n_clicks=0,
                        disabled=True,
                        style={"fontSize": "1rem", "padding": "0.6rem 1.2rem",
                               "background": "#315f8c", "color": "white", "border": 0,
                               "borderRadius": "7px", "marginBottom": "0.7rem"},
                    ),
                    dcc.Graph(id="dispatch-graph", figure=empty_dispatch_figure(),
                              config={"responsive": True, "displaylogo": False},
                              style={"width": "100%", "minHeight": "420px"}),
                    dcc.Graph(id="limiting-graph", figure=empty_limiting_figure(),
                              config={"responsive": True, "displaylogo": False},
                              style={"width": "100%"}),
                    html.P("Each cell is one day, coloured by what held methane "
                           "production back: an active equipment fault takes "
                           "precedence, then a missed target means the solar "
                           "resource ran short, while curtailed surplus means the "
                           "plant could not absorb what was available.",
                           style={"fontSize": "0.84rem", "color": "#60716c"}),
                    dcc.Graph(id="reactor-count-graph", figure=empty_reactor_count_figure(),
                              config={"responsive": True, "displaylogo": False},
                              style={"width": "100%", "minHeight": "360px"}),
                    html.H3("Compare test cases", style={"marginTop": "1.5rem"}),
                    html.P("Choose an aggregate metric calculated from the stored results."),
                    dcc.Dropdown(
                        id="comparison-metric",
                        options=[{"label": label, "value": key}
                                 for key, label in CASE_COMPARISON_METRICS.items()],
                        value="lcom_usd_per_kg_ch4", clearable=False,
                        style={"maxWidth": "560px", "marginBottom": "0.7rem"},
                    ),
                    html.Div([
                        html.Div([
                            html.Strong("Storage groups"),
                            dcc.Checklist(
                                id="comparison-storage-filter",
                                options=[{"label": f"All {LONG_STORAGE_LABELS[value]} cases", "value": value}
                                         for value in LONG_STORAGE_METHODS],
                                value=list(LONG_STORAGE_METHODS),
                                style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
                            ),
                        ]),
                        html.Div([
                            html.Strong("Strategy groups"),
                            dcc.Checklist(
                                id="comparison-strategy-filter",
                                options=[{"label": f"All {value.replace('_', ' ').title()} cases",
                                          "value": value}
                                         for value in OPERATING_STRATEGIES],
                                value=list(OPERATING_STRATEGIES),
                                style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"},
                            ),
                        ]),
                        html.Div([
                            html.Strong("Individual cases"),
                            dcc.Checklist(
                                id="comparison-case-filter",
                                options=[{
                                    "label": (f"{short_name.replace('_', ' ').title()} + "
                                              f"{LONG_STORAGE_LABELS[long_method]}"),
                                    "value": _result_key(short_name, long_method),
                                } for short_name in OPERATING_STRATEGIES
                                  for long_method in LONG_STORAGE_METHODS],
                                value=list(ALL_CASE_KEYS),
                                style={"display": "grid",
                                       "gridTemplateColumns": "repeat(auto-fit,minmax(220px,1fr))",
                                       "gap": "0.35rem 1rem"},
                            ),
                        ]),
                    ], style={"display": "grid", "gap": "0.65rem", "padding": "0.8rem",
                              "border": "1px solid #dbe0e8", "borderRadius": "8px",
                              "background": "#f8fafc"}),
                    dcc.Graph(id="comparison-bar", figure=empty_comparison_figure(),
                              config={"responsive": True, "displaylogo": False}),
                ]),
                dcc.Tab(label="Process flow diagram", children=[
                    dcc.Store(id="pfd-render-key"),
                    html.Div([
                        _control("Completed case", dcc.Dropdown(
                            id="pfd-case", options=[], value=None, clearable=False,
                            placeholder="Run a case first")),
                        _control("Result category", dcc.RadioItems(
                            id="pfd-category", value="imperfect", inline=True,
                            options=[{"label": "Perfect information", "value": "perfect"},
                                     {"label": "Imperfect forecast", "value": "imperfect"},
                                     {"label": "Imperfect with faults", "value": "imperfect_with_faults"}]))],
                        className="pfd-controls"),
                    html.P("Run a case to populate the stream labels.", id="pfd-status"),
                    html.P("Flow is averaged over the full simulation period, including shutdowns. "
                           "An asterisk marks an assumed temperature or pressure; N/A means the condition is not modelled. "
                           "All flows are totals across the installed trains.", id="pfd-basis", className="pfd-basis"),
                    html.Iframe(id="pfd-frame", srcDoc=render_pfd(), title="Process flow diagram with average stream conditions",
                                sandbox="", className="pfd-frame"),
                    _section("Stream values & calculation basis", html.Div(id="pfd-stream-table"), "#315f8c"),
                ]),
                dcc.Tab(label="Sizing", children=[
                    html.Div(
                        "Run a case to populate the equipment sizing register.",
                        id="sizing-content", style={"padding": "0.5rem 0"},
                    ),
                ]),
                dcc.Tab(label="European showcase", children=[
                    showcase_demo_panel(),
                    html.P(["Click dots or faint map points to select locations; click again to deselect. ",
                            _info("Runs use the current Single site settings and saved system parameters, with one base case per location. Comparison dots show fault-free imperfect results, or perfect results in Perfect only mode. Results accumulate until the app restarts, including across page refreshes.")]),
                    _section("Current Single site settings", html.Div(id="showcase-settings"), "#315f8c"),
                    _section("Save / load showcase", [
                        html.Button("Save showcase results", id="save-showcase-results"),
                        dcc.Upload(id="load-showcase-results", children=html.Button("Load result file"),
                                   accept=".json,.gz", multiple=False, max_size=MAX_FILE_BYTES),
                        html.P("Save map metrics, run settings and summary outputs. Loading merges locations into this map and replaces matching location names. Showcase files do not contain hourly dispatch traces."),
                        html.Div([
                            dcc.Input(id="showcase-save-path", type="text",
                                      value=str(DEFAULT_SAVE_DIR), debounce=True,
                                      style={"width": "min(420px, 100%)", "height": "34px"}),
                            html.Button("Save to disk", id="save-showcase-to-disk",
                                        n_clicks=0),
                        ], style={"display": "flex", "gap": "0.6rem", "flexWrap": "wrap",
                                  "alignItems": "center", "marginTop": "0.4rem"}),
                        html.Small(
                            "Writes the same file directly to this folder on the machine "
                            "running the app, bypassing the browser download entirely. "
                            "Relative paths are taken from the project directory.",
                            style={"color": "#60716c"},
                        ),
                        html.Div(id="showcase-file-status", role="status"),
                    ], "#315f8c"),
                    dcc.Store(id="showcase-selection", data=[]),
                    dcc.Store(id="showcase-results", data=[]),
                    dcc.Store(id="showcase-job"),
                    dcc.Interval(id="showcase-poll", interval=1000),
                    html.Div(id="showcase-selected-label"),
                    html.Div([
                        html.Div(id="showcase-ninja-usage", style={"fontSize": "0.82rem", "color": "#40534d"}),
                        html.Div(id="showcase-ninja-warning", role="status",
                                 style={"color": "#a21d1d", "fontSize": "0.82rem", "fontWeight": 600}),
                    ], style={"margin": "0.6rem 0"}),
                    showcase_weather_controls(),
                    showcase_mirror_controls(),
                    html.Button("Run 0 cases", id="run-showcase", n_clicks=0, disabled=True),
                    html.Button("Clear selection", id="clear-showcase", n_clicks=0),
                    html.Div(id="showcase-status", role="status"),
                    html.Div([
                        dcc.RadioItems(
                            id="map-metric", options=metric_options(False),
                            value=DEFAULT_MAP_METRIC, inline=True,
                            style={"display": "flex", "flexWrap": "wrap",
                                   "gap": "0.15rem 1.1rem"},
                        ),
                        dcc.Checklist(
                            id="show-advanced-metrics",
                            options=[{"label": " Show advanced metrics",
                                      "value": "advanced"}],
                            value=[], style={"marginTop": "0.5rem",
                                             "fontSize": "0.85rem"},
                        ),
                        dcc.Checklist(
                            id="hide-coordinate-grid",
                            options=[{"label": " Hide the coordinate grid — stops it "
                                               "catching the hover. Result dots stay "
                                               "clickable; picking a new location needs "
                                               "the grid back.",
                                      "value": "hide"}],
                            value=[], persistence=True, persistence_type="local",
                            style={"marginTop": "0.3rem", "fontSize": "0.85rem"},
                        ),
                    ], style={"margin": "0.6rem 0"}),
                    dcc.Graph(id="showcase-map", figure=build_showcase_map(clickable=True)),
                    html.Div(id="map-detail"),
                ]),
                dcc.Tab(label="System & economic parameters", children=[
                    html.Div([
                        html.H2("System and economic parameters",
                                style={"marginBottom": "0.35rem"}),
                        html.P([
                            "Edit the assumptions below, then select ",
                            html.Strong("Save parameters"),
                            ". New runs use the saved snapshot. Scenario inputs on the Single site tab ",
                            "(location, farm size, strategy, f_xOCP factors, initial SOC, and fault distributions) ",
                            "take precedence over their displayed defaults.",
                        ], style={"maxWidth": "900px", "color": "#40534d"}),
                        html.Div([
                            html.Button(
                                "Save parameters", id="save-assumptions", n_clicks=0,
                                style={"fontSize": "1rem", "padding": "0.6rem 1.2rem",
                                       "background": "#176b55", "color": "white", "border": 0,
                                       "borderRadius": "7px"},
                            ),
                            html.Span("Defaults loaded.", id="assumptions-save-status",
                                      style={"color": "#40534d"}),
                        ], style={"display": "flex", "gap": "0.8rem", "alignItems": "center",
                                  "margin": "1rem 0", "flexWrap": "wrap"}),
                        build_assumptions_table(),
                        html.P(
                            "Economic inputs remain illustrative placeholders even when edited.",
                            style={"color": "#8a4b08", "fontWeight": 600},
                        ),
                    ], style={"padding": "0.4rem 0"}),
                ]),
                dcc.Tab(label="Citations", children=[citations_panel()]),
            ]),
        ],
        style={"maxWidth": "1280px", "margin": "auto", "padding": "1rem", "fontFamily": "Arial, sans-serif", "color": "#17202a"},
    )

    @app.callback(
        Output("results-download", "data"), Output("run-job", "data", allow_duplicate=True),
        Output("site-file-status", "children"), Output("showcase-file-status", "children"),
        Output("demo-status", "children"), Output("showcase-demo-status", "children"),
        Input("save-site-results", "n_clicks"), Input("save-showcase-results", "n_clicks"),
        Input("save-showcase-to-disk", "n_clicks"),
        Input("load-site-results", "contents"), Input("load-showcase-results", "contents"),
        Input("load-demo", "n_clicks"),
        Input("load-demo-showcase-parallel", "n_clicks"),
        Input("load-demo-showcase-single", "n_clicks"),
        State("run-job", "data"), State("showcase-save-path", "value"),
        prevent_initial_call=True,
        running=[
            (Output("demo-progress", "children"), "Working, this takes about ten seconds...", ""),
            (Output("showcase-demo-progress", "children"),
             "Working, this takes about ten seconds...", ""),
            (Output("load-demo", "disabled"), True, False),
            (Output("load-demo-showcase-parallel", "disabled"), True, False),
            (Output("load-demo-showcase-single", "disabled"), True, False),
        ],
    )
    def result_file_action(_site_click, _map_click, _disk_click, site_upload,
                           map_upload, _demo_click, _parallel_click, _single_click,
                           job_data, save_path):
        trigger = ctx.triggered_id
        to_disk = trigger == "save-showcase-to-disk"
        on_map = to_disk or trigger in {"save-showcase-results", "load-showcase-results"}
        is_demo = trigger == "load-demo"
        variant = next((name for name in DEMO_SHOWCASES
                        if trigger == f"load-demo-showcase-{name}"), None)
        is_showcase_demo = variant is not None

        def response(message, download=no_update, job=no_update):
            # Every entry point owns the status line beside it, so no message -
            # including a failure raised below - lands in a panel out of view.
            return (download, job,
                    message if not (on_map or is_demo or is_showcase_demo) else no_update,
                    message if on_map else no_update,
                    message if is_demo else no_update,
                    message if is_showcase_demo else no_update)

        try:
            if is_demo or is_showcase_demo:
                # The showcase tab loads only the map half, so its button cannot
                # silently replace a single-site run the user is still looking at.
                loaded, restored, job = _load_demo_bundle(
                    DEMO_FILES if is_demo else (DEMO_SHOWCASES[variant][0],))
                if not loaded:
                    return response("No demo bundle is present. Build one with "
                                    "python scripts/build_demo.py.")
                nudge = ("Open European showcase for the map, or Single site to pick "
                         "the completed case and plot its dispatch." if is_demo else
                         f"Each site ran with {DEMO_SHOWCASES[variant][1]}. The map "
                         "below is populated; choose a metric to recolour it.")
                return response(
                    "Loaded " + " and ".join(loaded) + f", and restored {restored} "
                    f"cached weather years. {nudge}",
                    job=job if job is not None else no_update)
            if trigger in {"save-site-results", "save-showcase-results",
                           "save-showcase-to-disk"}:
                if on_map:
                    with RUN_JOBS_LOCK:
                        payload = list(SHOWCASE_RESULTS.values())
                    if not payload:
                        return response("Run or load showcase locations before saving.")
                    kind = "showcase"
                    # Carry the site label so a missing cache is reported by name
                    # rather than by its opaque hash.
                    sources = [{**(row.get("source_metadata") or {}),
                                "site": row.get("site")} for row in payload]
                else:
                    job = _job_snapshot(job_data["job_id"]) if job_data else None
                    if not job or job.get("status") != "complete" or not job.get("results"):
                        return response("Complete or load a single-site run before saving.")
                    kind = "single_site"
                    payload = {key: job[key] for key in JOB_FIELDS if key in job}
                    sources = [job.get("source_metadata", {})]
                weather, missing = collect_weather_cache(sources)
                contents = export_results(kind, payload, weather)
                # Timestamped so repeated saves land as distinct files rather than
                # "(1)", "(2)" copies in the browser's download folder.
                stamp = datetime.now().strftime("%Y%m%d-%H%M")
                name = f"solar-balancer-{kind}-{stamp}.json.gz"
                shortfall = ""
                if missing:
                    shortfall = (" Weather could not be bundled for "
                                 + "; ".join(missing)
                                 + " because it is no longer in the cache, so those "
                                 "locations will need re-fetching after loading.")
                if to_disk:
                    # Written by the server process, so this lands on the machine
                    # running the app rather than going through the browser.
                    folder = Path(save_path or DEFAULT_SAVE_DIR).expanduser()
                    folder.mkdir(parents=True, exist_ok=True)
                    target = (folder / name).resolve()
                    target.write_bytes(contents)
                    return response(
                        f"Wrote {len(contents) / 1e6:.1f} MB to {target} "
                        f"({len(weather)} cached weather years).{shortfall}"
                    )
                download = dcc.send_bytes(contents, name)
                return response(
                    f"Saved results with {len(weather)} cached weather years."
                    + shortfall, download)
            upload = map_upload if on_map else site_upload
            if not upload:
                return response("Choose a Solar Balancer result file.")
            if len(upload) > MAX_FILE_BYTES * 4 // 3 + 1024:
                raise ValueError("Uploaded file is too large.")
            raw = base64.b64decode(upload.split(",", 1)[1], validate=True)
            kind, payload, weather = import_results(raw)
            if kind == "single_site":
                previous = _job_snapshot(job_data["job_id"]) if job_data else None
                if previous and previous.get("status") in {"queued", "running"}:
                    return response("Wait for the current single-site run to finish before loading results.")
            restored = restore_weather_cache(weather)
            if kind == "showcase":
                with RUN_JOBS_LOCK:
                    SHOWCASE_RESULTS.update({row["site"]: row for row in payload})
                return response(f"Loaded {len(payload)} showcase locations and restored {restored} cached weather years.")
            job_id = _register_loaded_results(payload)
            return response(
                f"Loaded {len(payload['results'])} cases and restored {restored} cached weather years. "
                "Select a completed case to view results and load its dispatch plot.", job={"job_id": job_id})
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            return response(f"Could not save/load results: {exc}")

    @app.callback(
        Output("showcase-ninja-usage", "children"), Output("showcase-ninja-warning", "children"),
        Input("ninja-usage", "children"), Input("ninja-api-key-warning", "children"),
    )
    def mirror_showcase_ninja_status(usage, warning):
        return usage, warning

    @app.callback(
        Output("ninja-usage", "children"),
        Input("ninja-usage-poll", "n_intervals"),
        Input("weather-source", "value"), Input("ninja-api-key", "value"),
    )
    def show_ninja_usage(_, weather_source, entered_key):
        token = _resolve_ninja_api_key(entered_key)
        if not token:
            return "Ninja quota estimate: enter an API key."
        usage = usage_snapshot(token)
        label = f"Ninja: ~{usage['remaining']}/{usage['limit']} calls remaining (last hour)"
        if usage["retry_seconds"]:
            label += f" · Server retry in {usage['retry_seconds']}s"
        elif usage["next_release_seconds"]:
            minutes, seconds = divmod(usage["next_release_seconds"], 60)
            label += f" · Oldest call expires in {minutes}m {seconds:02d}s"
        if weather_source != "ninja":
            label += " · Synthetic uses no calls"
        return html.Span([label, _info(
            "Estimate assumes 50 requests per rolling hour and counts attempts made by this "
            "running app for this key, including retries and failed requests. Cached weather "
            "uses no calls. Usage elsewhere is unknown; restarting the app clears the estimate. "
            "This is not Ninja's reported remaining quota or an authoritative reset time."
        )])

    @app.callback(
        Output("ninja-api-key-warning", "children"),
        Input("weather-source", "value"), Input("ninja-api-key", "value"),
    )
    def warn_missing_ninja_api_key(weather_source, entered_key):
        # _resolve_ninja_api_key checks the configured .env token before the box, so
        # this stays quiet whenever either source can supply a key.
        if weather_source != "ninja" or _resolve_ninja_api_key(entered_key):
            return ""
        # Names both alternatives the way the weather-source radio labels them, so
        # the way out of the warning is findable rather than merely implied.
        return ("A Renewables.ninja API key is required for loading external weather "
                "data. Supply a key, or use synthetic weather data (Offline "
                "synthetic) or pre-loaded profiles (Cached only).")

    @app.callback(
        Output("calibration-controls", "style"),
        Input("capacity-setting-mode", "value"),
    )
    def show_calibration_search(capacity_setting_mode):
        return {**CALIBRATION_BLOCK_STYLE,
                "display": "grid" if capacity_setting_mode == "auto" else "none"}

    @app.callback(
        Output("information-mode", "value"), Output("fault-mode-note", "children"),
        Input("information-mode", "value"), Input("fault-enabled", "value"),
        prevent_initial_call=True,
    )
    def keep_faults_and_information_mode_consistent(information_mode, fault_enabled):
        """Faults need a forecast to be measured against, so run_case refuses them in
        perfect-information-only mode. Rather than accept the setting and discard it,
        asking for faults asks for the mode that can deliver them."""
        if "enabled" not in (fault_enabled or []):
            return no_update, ""
        if information_mode != "perfect_only":
            # Either already correct, or this is the echo of the switch made below,
            # whose explanation must survive its own re-trigger.
            return no_update, no_update
        if ctx.triggered_id == "fault-enabled":
            return "comparison", (
                "Information mode switched to Perfect vs imperfect, because faults are "
                "measured against a forecast and a perfect-information-only run has "
                "none to compare with."
            )
        return no_update, (
            "Faults will not be applied while Information mode is Perfect only: they "
            "are measured against a forecast. Switch to Perfect vs imperfect to "
            "include them."
        )

    @app.callback(
        Output("result-category", "value"),
        Input("fault-enabled", "value"), Input("information-mode", "value"),
        State("result-category", "value"),
    )
    def select_default_result_category(fault_enabled, information_mode, current):
        if information_mode == "perfect_only":
            return "perfect"
        if "enabled" in (fault_enabled or []):
            return "imperfect_with_faults"
        return "imperfect" if current == "imperfect_with_faults" else (current or "imperfect")

    @app.callback(
        Output("f-ocp", "min"), Output("f-ocp", "max"), Output("f-ocp", "step"),
        Output("f-ocp", "marks"), Output("f-ocp", "value"),
        Output("f-socp", "min"), Output("f-socp", "max"), Output("f-socp", "step"),
        Output("f-socp", "marks"), Output("f-socp", "value"),
        Output("factor-range-status", "children"),
        Input("factor-range-min", "value"), Input("factor-range-max", "value"),
        Input("factor-range-step", "value"),
        Input("calibrated-factors", "data"),
        State("f-ocp", "value"), State("f-socp", "value"),
    )
    def update_factor_slider_scale(minimum, maximum, increment, calibrated_factors,
                                   f_ocp, f_socp):
        try:
            minimum, maximum, increment, marks = _slider_scale(
                minimum, maximum, increment
            )
        except (TypeError, ValueError) as exc:
            return tuple([no_update] * 10 + [f"Invalid slider scale: {exc}."])
        selected_values = (f_ocp, f_socp)
        auto_updated = ctx.triggered_id == "calibrated-factors" and calibrated_factors
        if auto_updated:
            selected_values = (
                calibrated_factors["f_ocp"], calibrated_factors["f_socp"],
            )
        values = []
        for value, fallback in zip(selected_values, (0.20, 1.0)):
            current = fallback if value is None else float(value)
            values.append(min(maximum, max(minimum, current)))
        slider_outputs = []
        for value in values:
            slider_outputs.extend((minimum, maximum, increment, marks, value))
        status = (f"Scale: {minimum:g} to {maximum:g} in steps of {increment:g}, "
                  "applied to f_OCP and f_SOCP only.")
        if auto_updated:
            status += (
                " Auto-calibrated values loaded after "
                f"{calibrated_factors['evaluations']} perfect-information evaluations."
            )
        return tuple(slider_outputs + [status])

    @app.callback(
        Output("slow-calibration-warning", "children"),
        Input("reactor-scheduling-mode", "value"),
        Input("capacity-setting-mode", "value"),
    )
    def warn_slow_daily_auto_calibration(reactor_scheduling_mode, capacity_setting_mode):
        return _slow_calibration_warning(
            reactor_scheduling_mode, capacity_setting_mode,
        )

    @app.callback(Output("showcase-settings", "children"),
                  *[Input(key, "value") for key, _ in SHOWCASE_SETTING_FIELDS])
    def update_showcase_settings(*values):
        return showcase_settings_summary(values)

    @app.callback(Output("showcase-map", "figure"), Input("map-metric", "value"),
                  Input("showcase-results", "data"), Input("showcase-selection", "data"),
                  Input("hide-coordinate-grid", "value"))
    def update_map(metric, records, selected, hide_grid=None):
        return build_showcase_map(metric, records, selected, clickable=True,
                                  show_grid="hide" not in (hide_grid or []))

    @app.callback(Output("map-metric", "options"), Output("map-metric", "value"),
                  Input("show-advanced-metrics", "value"), State("map-metric", "value"))
    def update_metric_options(advanced, metric):
        show_advanced = "advanced" in (advanced or [])
        options = metric_options(show_advanced)
        # Collapsing the advanced list must not leave an unselectable metric behind.
        if not show_advanced and metric not in SIMPLE_METRIC_META:
            metric = DEFAULT_MAP_METRIC
        return options, metric

    for _main_id, _twin_id in SHOWCASE_MIRRORED_CONTROLS:
        # Dash supports a callback whose inputs and outputs are the same properties,
        # which is how two controls are kept in step without a circular dependency
        # error. The trigger decides which side is authoritative for this update.
        @app.callback(Output(_main_id, "value"), Output(_twin_id, "value"),
                      Input(_main_id, "value"), Input(_twin_id, "value"),
                      prevent_initial_call=True)
        def sync_showcase_control(main_value, twin_value, main_id=_main_id):
            value = main_value if ctx.triggered_id == main_id else twin_value
            return value, value

    @app.callback(Output("showcase-selection", "data"),
                  Input("showcase-map", "clickData"), Input("clear-showcase", "n_clicks"),
                  State("showcase-selection", "data"), prevent_initial_call=True)
    def select_showcase(click_data, _, selected):
        return [] if ctx.triggered_id == "clear-showcase" else toggle_showcase_location(click_data, selected)

    @app.callback(Output("run-showcase", "children"), Output("run-showcase", "disabled"),
                  Output("showcase-selected-label", "children"),
                  Output("showcase-status", "children"), Output("showcase-results", "data"),
                  Input("showcase-poll", "n_intervals"), Input("showcase-selection", "data"),
                  Input("showcase-job", "data"), State("showcase-results", "data"))
    def poll_showcase(_, selected, job_data, records):
        selected = selected or []
        job = _job_snapshot(job_data["job_id"]) if job_data else None
        running = bool(job and job["status"] in {"queued", "running"})
        status = "Select locations, then run their cases."
        with RUN_JOBS_LOCK:
            current_records = [dict(row) for row in SHOWCASE_RESULTS.values()]
        updated = current_records if current_records != (records or []) else no_update
        if job:
            status = html.Div([job["messages"][-1],
                               html.Ul([html.Li(error) for error in job.get("errors", [])])])
        elif job_data:
            status = "Run details have expired. Completed map results are retained; you can run again."
        return (f"Run {len(selected)} cases", running or not selected,
                ", ".join(row["site"] for row in selected) or "No locations selected.", status, updated)

    @app.callback(
        Output("saved-assumptions", "data"),
        Output("assumptions-save-status", "children"),
        Input("save-assumptions", "n_clicks"),
        State({"type": "assumption-input", "key": ALL}, "id"),
        State({"type": "assumption-input", "key": ALL}, "value"),
        prevent_initial_call=True,
    )
    def save_assumptions(_, input_ids, values):
        config = {item["key"]: value for item, value in zip(input_ids, values)}
        blank = next((key for key, value in config.items() if value is None), None)
        if blank:
            return no_update, f"Not saved: {blank} cannot be blank."
        try:
            plant_values = _assumption_section(
                config, "plant", PlantParameters(), PLANT_ASSUMPTION_META
            )
            plant_values.update(_assumption_section(
                config, "plant", PlantParameters(), PHYSICAL_PROPERTY_ASSUMPTION_META
            ))
            PlantParameters(**plant_values)
            ThermalParameters(**_assumption_section(
                config, "thermal", ThermalParameters(), THERMAL_ASSUMPTION_META
            ))
            StrategyConfig(**_assumption_section(
                config, "strategy", StrategyConfig(), STRATEGY_ASSUMPTION_META
            ))
            weather = _assumption_section(
                config, "weather", WeatherConfig(0, 0), WEATHER_ASSUMPTION_META
            )
            if weather["training_years"] < 1 or weather["evaluation_years"] < 1:
                raise ValueError("weather training_years and evaluation_years must be positive")
            synthetic = _assumption_section(
                config, "synthetic", SyntheticWeatherParameters(),
                SYNTHETIC_WEATHER_ASSUMPTION_META,
            )
            SyntheticWeatherParameters(**synthetic)
            if (int(config["synthetic.training_years"]) < 1
                    or int(config["synthetic.evaluation_years"]) < 1):
                raise ValueError("synthetic training_years and evaluation_years must be positive")
            for name in ("seasonal_period_days", "diurnal_period_hours"):
                if synthetic[name] <= 0:
                    raise ValueError(f"synthetic.{name} must be positive")
            if synthetic["cloud_variability"] < 0:
                raise ValueError("synthetic.cloud_variability cannot be negative")
            for name in ("clear_sky_index", "atmospheric_transmittance"):
                if not 0 < synthetic[name] <= 1:
                    raise ValueError(f"synthetic.{name} must lie above zero and at or below one")
            if not 0 <= synthetic["axial_tilt_deg"] < 90:
                raise ValueError("synthetic.axial_tilt_deg must lie between 0 and 90 degrees")
            if not 0 <= synthetic["array_tilt_deg"] <= 90:
                raise ValueError("synthetic.array_tilt_deg must lie between 0 and 90 degrees")
            if not 0 <= synthetic["minimum_sun_elevation_sine"] < 1:
                raise ValueError("synthetic.minimum_sun_elevation_sine must lie between 0 and 1")
            for key, value in config.items():
                if any(token in key for token in (
                    "efficiency", "initial_soc_fraction",
                    "maximum_capacity_factor",
                )) and not 0 <= float(value) <= 1:
                    raise ValueError(f"{key} must lie between 0 and 1")
                if "self_discharge_fraction_per_h" in key and float(value) < 0:
                    raise ValueError(f"{key} cannot be negative")
            for name in PHYSICAL_PROPERTY_ASSUMPTION_META:
                if float(plant_values[name]) <= 0:
                    raise ValueError(f"plant.{name} must be positive")
            costs = _economic_costs_from_assumptions(config)
            if costs["financial"]["project_life_years"] <= 0:
                raise ValueError("financial.project_life_years must be positive")
            if costs["financial"]["real_discount_rate"] < 0:
                raise ValueError("financial.real_discount_rate cannot be negative")
            for key, value in config.items():
                if key.startswith("economic.") and isinstance(value, (int, float)) and value < 0:
                    raise ValueError(f"{key} cannot be negative")
        except (TypeError, ValueError, KeyError) as exc:
            return no_update, f"Not saved: {exc}."
        return config, "Saved. New runs will use these parameter values."

    @app.callback(
        Output({"type": "assumption-input", "key": ALL}, "value"),
        Input("saved-assumptions", "data"),
        State({"type": "assumption-input", "key": ALL}, "id"),
    )
    def load_saved_assumptions(saved, input_ids):
        config = default_assumption_config()
        config.update(saved or {})
        return [config[item["key"]] for item in input_ids]

    @app.callback(
        Output("weather-source-showcase", "value", allow_duplicate=True),
        Output("showcase-selection", "data", allow_duplicate=True),
        Output("cached-weather-status-showcase", "children"),
        Input("load-cached-weather-showcase", "n_clicks"), prevent_initial_call=True)
    def load_cached_weather_for_showcase(_):
        # Switch to cached-only and select exactly the sites that can run from
        # disk, so there is no hunting for the right dots on the map.
        entries = cached_site_entries()
        runnable = [entry for entry in entries if entry["runnable"]]
        if not runnable:
            return no_update, no_update, (
                "No site has a complete span of cached weather yet. Run once with "
                "Renewables.ninja to populate the cache."
            )
        selection = [{"site": entry["site"], "lat": entry["lat"], "lon": entry["lon"]}
                     for entry in runnable]
        skipped = [entry["site"] for entry in entries if not entry["runnable"]]
        return "cached", selection, html.Div([
            html.Div(f"Cached-only mode selected and {len(selection)} sites selected "
                     "on the map. No API calls can be spent."),
            html.Div(", ".join(entry["site"] for entry in runnable),
                     style={"color": "#60716c"}),
            *([html.Div(f"Skipped (incomplete cache): {', '.join(skipped)}",
                        style={"color": "#8a4b08"})] if skipped else []),
        ])

    @app.callback(
        Output("weather-source", "value"), Output("cached-site", "options"),
        Output("cached-weather-status", "children"),
        Input("load-cached-weather", "n_clicks"), prevent_initial_call=True)
    def load_cached_weather(_):
        options = cached_site_options()
        if not options:
            return no_update, [], ("No weather is cached yet. Run once with "
                                   "Renewables.ninja to populate the cache.")
        return "cached", options, (
            f"{len(options)} cached sites available. Cached-only mode never contacts "
            "Renewables.ninja, so no API calls can be spent. Pick a site to jump to it."
        )

    @app.callback(
        Output("lat", "value", allow_duplicate=True),
        Output("lon", "value", allow_duplicate=True),
        Output("latest-year", "value", allow_duplicate=True),
        Input("cached-site", "value"), prevent_initial_call=True)
    def jump_to_cached_site(selection):
        if not selection:
            raise PreventUpdate
        site = json.loads(selection)
        return site["lat"], site["lon"], site["latest"]

    @app.callback(Output("lat", "value"), Output("lon", "value"),
                  Input("location-picker", "clickData"), prevent_initial_call=True)
    def pick_location(click_data):
        point = click_data["points"][0]
        coordinate = point.get("customdata")
        if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
            return float(coordinate[0]), float(coordinate[1])
        return float(point["lat"]), float(point["lon"])

    @app.callback(Output("location-picker", "figure"),
                  Input("lat", "value"), Input("lon", "value"))
    def update_location_picker(lat, lon):
        return build_location_picker(float(lat), float(lon))

    @app.callback(Output("map-detail", "children"), Input("showcase-map", "clickData"),
                  Input("showcase-results", "data"))
    def show_map_detail(click_data, records):
        if not click_data or not click_data.get("points"):
            return "Click a site for details."
        point = click_data["points"][0]
        site = point.get("hovertext") or f"{point['lat']:.2f}, {point['lon']:.2f}"
        row = showcase_data(records).loc[lambda frame: frame["site"] == site]
        if row.empty:
            return "Run this location to populate its information dot."
        item = row.iloc[0]
        def show(key, spec, missing="N/A"):
            value = item.get(key)
            return format(value, spec) if pd.notna(value) else missing

        return html.Div([
            html.H3(site), html.P(f"Data status: {item.data_status}"),
            html.P("Annual methane (perfect / imperfect / with faults): "
                   + " / ".join(show(key, ",.0f") for key in
                                ("ch4_perfect", "ch4_imperfect", "ch4_faults")) + " kg/y"),
            html.P("Illustrative LCOM (perfect / imperfect / with faults): "
                   + " / ".join("$" + show(key, ",.2f") for key in
                                ("lcom_perfect", "lcom_imperfect", "lcom_faults")) + "/kg"),
            html.P(f"Forecast + fault production ratio: {show('production_ratio', '.3f')}"),
            html.P("No-storage baseline production ratio: "
                   f"{show('baseline_production_ratio', '.3f')}"
                   " — the same plant run flat out with no storage and the same "
                   "faults, as a fraction of what this plant realises"),
            html.P(f"Forecast + fault cost ratio: {show('forecast_cost_ratio', '.3f')}"
                   + f" — {item.forecast_cost_ratio_basis}"),
            html.P(f"Storage: {show('short_storage_capacity_mwh', ',.1f')} MWh short / "
                   f"{show('long_storage_capacity_mwh', ',.1f')} MWh long / "
                   f"{show('methane_storage_capacity_kg', ',.0f')} kg CH4"),
            html.P(f"Cyclic energy deficit: {show('annual_balance_deficit_mwh', ',.2f')} MWh/y"),
            html.P(f"Utilisation: {show('plant_utilisation', '.1%')}; "
                   f"curtailment: {show('curtailment_fraction', '.1%')}"),
        ])

    @app.callback(
        Output("run-job", "data"),
        Output("showcase-job", "data"),
        Input("run-specific", "n_clicks"), Input("run-all", "n_clicks"),
        Input("run-showcase", "n_clicks"),
        State("lat", "value"), State("lon", "value"),
        State("farm-mw", "value"), State("weather-source", "value"),
        State("ninja-api-key", "value"), State("latest-year", "value"),
        State("reactor-count", "value"),
        State("reactor-scheduling-mode", "value"),
        State("information-mode", "value"),
        State("short-strategy", "value"), State("long-storage", "value"),
        State("product-storage", "value"),
        State("capacity-setting-mode", "value"),
        State("f-ocp", "value"), State("f-socp", "value"),
        State("air-hx-effectiveness", "value"),
        State("calibration-range", "value"), State("calibration-step", "value"),
        State("initial-soc", "value"),
        State("fault-enabled", "value"), State("fault-seed", "value"),
        State("fault-battery-months", "value"), State("fault-battery-duration", "value"),
        State("fault-battery-capacity", "value"),
        State("fault-hydrogen-months", "value"), State("fault-hydrogen-duration", "value"),
        State("fault-hydrogen-capacity", "value"),
        State("fault-sabatier-months", "value"), State("fault-sabatier-duration", "value"),
        State("fault-sabatier-capacity", "value"),
        State("fault-dac-months", "value"), State("fault-dac-duration", "value"),
        State("fault-dac-capacity", "value"),
        State("saved-assumptions", "data"),
        State({"type": "investigate", "parameter": ALL}, "value"),
        State("showcase-selection", "data"), State("showcase-job", "data"),
        prevent_initial_call=True,
    )
    def start_model(_, __, ___, lat, lon, farm_mw, weather_source, ninja_api_key,
                    latest_year, reactor_count, reactor_scheduling_mode, information_mode,
                    short_name, long_method, product_storage, capacity_setting_mode,
                    f_ocp, f_socp, air_hx_effectiveness,
                    calibration_range, calibration_increment,
                    initial_soc_fraction, fault_enabled, fault_seed,
                    battery_months, battery_duration, battery_capacity,
                    hydrogen_months, hydrogen_duration, hydrogen_capacity,
                    sabatier_months, sabatier_duration, sabatier_capacity,
                    dac_months, dac_duration, dac_capacity, saved_assumptions, investigations=None,
                    showcase_selection=None, showcase_job=None):
        job_id = uuid4().hex
        is_showcase = ctx.triggered_id == "run-showcase"
        previous = _job_snapshot(showcase_job["job_id"]) if showcase_job else None
        if is_showcase and (not showcase_selection or (
                previous and previous["status"] in {"queued", "running"})):
            return no_update, no_update
        run_scope = "investigate" if ctx.triggered_id == "run-all" else "specific"
        investigate = [key for values in (investigations or []) for key in values]
        if run_scope == "investigate" and not investigate:
            return no_update, no_update
        parameters = {
            "lat": lat, "lon": lon, "farm_mw": farm_mw, "weather_source": weather_source,
            "ninja_api_key": ninja_api_key,
            "latest_year": latest_year, "reactor_count": reactor_count,
            "reactor_scheduling_mode": reactor_scheduling_mode,
            "information_mode": information_mode,
            "investigate": investigate, "run_scope": run_scope, "short_name": short_name, "long_method": long_method,
            "product_storage": "enabled" in (product_storage or []),
            "capacity_setting_mode": capacity_setting_mode,
            "f_ocp": f_ocp, "f_socp": f_socp,
            "air_hx_effectiveness": air_hx_effectiveness,
            "initial_soc_fraction": initial_soc_fraction,
            # Kept as three scalars because that is what calibrate_capacity_factors
            # takes; the interface offers them as one range plus a grid.
            "factor_minimum": (calibration_range or [0.0, 2.0])[0],
            "factor_maximum": (calibration_range or [0.0, 2.0])[1],
            "factor_increment": calibration_increment,
            "faults_enabled": "enabled" in (fault_enabled or []),
            "fault_seed": fault_seed,
            "fault_config": {
                "battery": {"months": battery_months, "duration_h": battery_duration,
                            "capacity_fraction": battery_capacity},
                "hydrogen": {"months": hydrogen_months, "duration_h": hydrogen_duration,
                             "capacity_fraction": hydrogen_capacity},
                "sabatier": {"months": sabatier_months, "duration_h": sabatier_duration,
                             "capacity_fraction": sabatier_capacity},
                "dac": {"months": dac_months, "duration_h": dac_duration,
                        "capacity_fraction": dac_capacity},
            },
            "assumptions": saved_assumptions,
        }
        with RUN_JOBS_LOCK:
            RUN_JOBS[job_id] = {"status": "queued", "messages": ["Run queued."],
                                "delivered_selection": None, "delivered_failure": False,
                                "delivered_comparison": None}
            completed = [key for key, value in RUN_JOBS.items()
                         if key != job_id and not value.get("internal")
                         and value.get("status") in {"complete", "failed"}]
            # Each completed all-case job now holds nine full-resolution result sets, so retain
            # only a small number of prior jobs to bound server memory.
            for old_job_id in completed[:-MAX_COMPLETED_JOBS]:
                RUN_JOBS.pop(old_job_id, None)
        if is_showcase:
            Thread(target=_execute_showcase_job,
                   args=(job_id, list(showcase_selection), parameters), daemon=True).start()
            return no_update, {"job_id": job_id}
        Thread(target=_execute_single_site_job, args=(job_id, parameters), daemon=True).start()
        return {"job_id": job_id}, no_update

    @app.callback(
        Output("save-size-estimate", "children"),
        Output("save-size-estimate", "style"),
        Input("run-job-poll", "n_intervals"),
        State("run-job", "data"),
    )
    def estimate_save_size(_, job_data):
        """Say how large a save would be before it is attempted, not after it fails."""
        base = {"fontSize": "0.85rem", "fontWeight": 600, "marginTop": "0.4rem"}
        if not job_data or not job_data.get("job_id"):
            return "", base
        job = _job_snapshot(job_data["job_id"])
        if job is None or job.get("status") != "complete" or not job.get("results"):
            return "", base
        summary = describe_export_size({key: job[key] for key in JOB_FIELDS
                                        if key in job})
        if summary is None:
            return "", base
        over = "over the" in summary
        near = "close to the limit" in summary
        colour = "#a21d1d" if over else ("#8a4b08" if near else "#60716c")
        return summary, {**base, "color": colour}

    @app.callback(
        Output("run-status", "children"), Output("metric-cards", "children"),
        Output("run-specific", "disabled"), Output("run-all", "disabled"),
        Output("load-dispatch", "disabled"),
        Output("calibrated-factors", "data"),
        Input("run-job-poll", "n_intervals"),
        Input("short-strategy", "value"), Input("long-storage", "value"),
        Input("capacity-setting-mode", "value"), Input("result-category", "value"),
        State("run-job", "data"),
        Input({"type": "investigate", "parameter": ALL}, "value"),
        Input("completed-case", "value"),
    )
    def poll_model(_, short_strategy, long_storage, capacity_setting_mode,
                   result_category, job_data, investigations=None, completed_case=None):
        run_all_disabled = capacity_setting_mode == "auto" or (investigations is not None and not any(investigations))
        if not job_data or not job_data.get("job_id"):
            return no_update, no_update, False, run_all_disabled, True, no_update
        job_id = job_data["job_id"]
        job = _job_snapshot(job_id)
        if job is None:
            return html.Div("Run status was lost; please start the case again.",
                            style={"color": "#a21d1d"}), [], False, run_all_disabled, True, no_update
        messages = job.get("messages", [])
        if job["status"] in {"queued", "running"}:
            visible_messages = messages[-10:]
            status = html.Div([
                html.Strong(visible_messages[-1] if visible_messages else "Working…"),
                html.Ul([html.Li(message) for message in visible_messages],
                        style={"marginTop": "0.5rem", "maxHeight": "13rem", "overflowY": "auto"}),
            ], style={"padding": "0.8rem", "background": "#f4f8f6",
                      "borderLeft": "4px solid #176b55"})
            return status, no_update, True, True, True, no_update
        if job["status"] == "failed":
            if job.get("delivered_failure"):
                return no_update, no_update, False, run_all_disabled, True, no_update
            _update_job(job_id, delivered_failure=True)
            status = html.Div([
                html.Strong(f"Run failed: {job.get('error', 'unknown error')}"),
                html.Details([html.Summary("Show run steps"),
                              html.Ul([html.Li(message) for message in messages])], open=True),
            ], style={"color": "#a21d1d"})
            return status, [], False, run_all_disabled, True, no_update
        selection = completed_case or _result_key(short_strategy, long_storage)
        if completed_case:
            short_strategy, long_storage = selection.split("|")[:2]
        delivery_key = f"{selection}|{result_category}"
        missing_selection = f"missing:{selection}"
        if job.get("delivered_selection") in {delivery_key, missing_selection}:
            return no_update, no_update, False, run_all_disabled, False, no_update
        result = job["results"].get(selection)
        if result is None:
            available = next(iter(job["results"]), None)
            message = (
                f"{selection} was not calculated by the selected-case run. "
                f"Available result: {available}. Select it or click Run selected case."
            )
            _update_job(job_id, delivered_selection=missing_selection)
            return html.Div(message), [], False, run_all_disabled, True, no_update
        information_mode = job.get("information_mode", "comparison")
        perfect_only = information_mode == "perfect_only"
        selected_category = "perfect" if perfect_only else result_category
        simulation, economics, category_label = _result_category_parts(
            result, selected_category,
        )
        if simulation is None or economics is None:
            return html.Div(f"Result {selection} is incomplete."), [], False, run_all_disabled, True, no_update
        feasibility = ("Cyclic storage balance closes." if result.sizing.feasible else
                       "Cyclic storage balance does not close; dispatch results remain valid.")
        deficit = (f"Average deficit="
                   f"{result.sizing.average_annual_balance_deficit_kwh / 1000:,.2f} MWh/year.")
        sizing_warning = " ".join(result.sizing.warnings)
        # The forecast-driven cases run a plant sized before the evaluation period, so
        # whether that plant closes its cycle on the weather it actually met is a
        # separate question from whether the perfect-information one does.
        built = getattr(result, "sizing_imperfect", None)
        if built is not None:
            realised = getattr(result, "imperfect_realised_deficit_kwh_per_year", 0.0)
            deficit += (
                " The plant sized on the training record was built with "
                f"{built.long_capacity_kwh / 1000:,.1f} MWh of long storage against the "
                f"perfect-information {result.sizing.long_capacity_kwh / 1000:,.1f} MWh, "
                + (f"and runs an average deficit of {realised / 1000:,.2f} MWh/year on "
                   "the evaluation weather." if realised > 1e-6
                   else "and still closes its cycle on the evaluation weather.")
            )
        run_summary = (f"{len(job['results'])} test cases calculated" if job.get("run_scope") in {"all", "investigate"}
                       else "Selected case calculated")
        optimized_factors = job.get("optimized_factors")
        calibration_text = ""
        if optimized_factors:
            calibration_text = (
                " Auto-calibration selected "
                f"f_OCP={optimized_factors['f_ocp']:g}, "
                f"f_SOCP={optimized_factors['f_socp']:g} after "
                f"{optimized_factors['evaluations']} perfect-information evaluations."
            )
            if optimized_factors["limit_reached"]:
                calibration_text += (
                    f" The {MAX_CAPACITY_CALIBRATION_EVALUATIONS}-evaluation limit was reached."
                )
        completion = (
            f"{run_summary}. Showing case {result.case_id}: "
            f"{job.get('case_labels', {}).get(selection, short_strategy + ' + ' + long_storage)}; category={category_label}; "
            f"source={job['source_metadata']['source']}. "
            f"{feasibility} {deficit}{calibration_text} {sizing_warning} {economics['cost_warning']} "
            "Click Load selected dispatch plot to render the full-resolution dispatch and reactor-count plots."
        )
        _update_job(job_id, delivered_selection=delivery_key)
        status = html.Div([
            html.Div(completion),
            html.Details([html.Summary("Show run steps"),
                          html.Ul([html.Li(message) for message in messages])]),
        ])
        cards = _result_cards(
            result, job.get("evaluation_period"), information_mode,
            result_category=selected_category,
            short_strategy=short_strategy, long_storage=long_storage,
        )
        if optimized_factors:
            cards.insert(0, _number_card(
                "Auto-calibrated factors",
                f"f_OCP={optimized_factors['f_ocp']:g}; "
                f"f_SOCP={optimized_factors['f_socp']:g}",
            ))
            start_lcom = optimized_factors.get("starting_lcom_usd_per_kg_ch4")
            best_lcom = optimized_factors.get("optimized_lcom_usd_per_kg_ch4")
            cards.insert(1, _number_card(
                "Calibration perfect LCOM",
                (f"${start_lcom:,.2f} → ${best_lcom:,.2f}/kg"
                 if start_lcom is not None and best_lcom is not None else "undefined"),
            ))
        return status, cards, False, run_all_disabled, False, optimized_factors or no_update

    @app.callback(
        Output("dispatch-graph", "figure"),
        Output("reactor-count-graph", "figure"),
        Output("limiting-graph", "figure"),
        Input("load-dispatch", "n_clicks"), Input("run-job", "data"),
        State("short-strategy", "value"), State("long-storage", "value"),
        State("result-category", "value"),
        State("completed-case", "value"),
        prevent_initial_call=True,
    )
    def load_dispatch_plot(_, job_data, short_strategy, long_storage, result_category, completed_case=None):
        if ctx.triggered_id == "run-job":
            message = "Calculations are running. Select a completed case and load its plots afterward."
            return (empty_dispatch_figure(message), empty_reactor_count_figure(message),
                    empty_limiting_figure(message))
        if not job_data or not job_data.get("job_id"):
            message = "Run a case before loading dispatch plots."
            return (empty_dispatch_figure(message), empty_reactor_count_figure(message),
                    empty_limiting_figure(message))
        job = _job_snapshot(job_data["job_id"])
        if job is None or job.get("status") != "complete":
            message = "The calculations have not completed yet."
            return (empty_dispatch_figure(message), empty_reactor_count_figure(message),
                    empty_limiting_figure(message))
        selection = completed_case or _result_key(short_strategy, long_storage)
        result = job["results"].get(selection)
        if result is None:
            message = "That combination was not calculated. Select an available case or run it first."
            return (empty_dispatch_figure(message), empty_reactor_count_figure(message),
                    empty_limiting_figure(message))
        perfect_only = job.get("information_mode", "comparison") == "perfect_only"
        selected_category = "perfect" if perfect_only else result_category
        simulation, _, information_label = _result_category_parts(result, selected_category)
        if simulation is None:
            message = "The selected dispatch result is unavailable."
            return (empty_dispatch_figure(message), empty_reactor_count_figure(message),
                    empty_limiting_figure(message))
        title = (
            f"Case {result.case_id} — {job['source_metadata']['source']} — "
            f"{job['evaluation_period']} — "
            f"{information_label} — "
            f"{job.get('case_labels', {}).get(selection, short_strategy + ' + ' + long_storage)}"
        )
        return (
            build_dispatch_figure(simulation, title),
            build_reactor_count_figure(simulation, f"{title} — reactor trains"),
            build_limiting_subsystem_figure(
                simulation, f"Limiting subsystem by day — {information_label}"),
        )

    @app.callback(
        Output("sizing-content", "children"),
        Input("run-job-poll", "n_intervals"),
        Input("short-strategy", "value"), Input("long-storage", "value"),
        Input("result-category", "value"),
        State("run-job", "data"),
        Input("completed-case", "value"),
    )
    def update_sizing_tab(_, short_strategy, long_storage, result_category, job_data, completed_case=None):
        if not job_data or not job_data.get("job_id"):
            return "Run a case to populate the equipment sizing register."
        job = _job_snapshot(job_data["job_id"])
        if job is None:
            return "The stored run is no longer available."
        if job.get("status") in {"queued", "running"}:
            return "Sizing is being calculated."
        if job.get("status") == "failed":
            return "The case calculation failed so no sizing register is available."
        selection = completed_case or _result_key(short_strategy, long_storage)
        short_strategy, long_storage = selection.split("|")[:2]
        result = job["results"].get(selection)
        if result is None:
            return "That combination was not calculated. Select an available case or run it first."
        perfect_only = job.get("information_mode", "comparison") == "perfect_only"
        selected_category = "perfect" if perfect_only else result_category
        simulation, _, information_label = _result_category_parts(result, selected_category)
        return html.Div([
            build_sizing_table(result, simulation, selected_category),
            dcc.Graph(
                figure=build_active_reactor_figure(
                    simulation,
                    f"Daily reactor count — {short_strategy.replace('_', ' ').title()} + "
                    f"{LONG_STORAGE_LABELS[long_storage]} — {information_label}",
                ),
                config={"responsive": True, "displaylogo": False},
                style={"marginTop": "1.2rem"},
            ),
            html.P(
                "Each point is the maximum integer reactor count reached during that UTC day; "
                "the dashed line is the forecast plan and the solid line is the realized count.",
                style={"color": "#40534d"},
            ),
        ])

    @app.callback(
        Output("comparison-bar", "figure"),
        Input("comparison-metric", "value"),
        Input("comparison-storage-filter", "value"),
        Input("comparison-strategy-filter", "value"),
        Input("comparison-case-filter", "value"),
        Input("run-job-poll", "n_intervals"),
        State("run-job", "data"),
    )
    def update_case_comparison(metric, selected_storage, selected_strategies,
                               selected_cases, _, job_data):
        if not job_data or not job_data.get("job_id"):
            return no_update
        job_id = job_data["job_id"]
        job = _job_snapshot(job_id)
        if job is None:
            return empty_comparison_figure("The stored run is no longer available.")
        if job.get("status") in {"queued", "running"}:
            if job.get("delivered_comparison") == "running":
                return no_update
            _update_job(job_id, delivered_comparison="running")
            return empty_comparison_figure("Calculations are still running.")
        if job.get("status") == "failed":
            return empty_comparison_figure("The case calculation failed.")
        token = (
            f"complete:{metric}:"
            f"{','.join(sorted(selected_storage or []))}:"
            f"{','.join(sorted(selected_strategies or []))}:"
            f"{','.join(sorted(selected_cases or []))}"
        )
        if job.get("delivered_comparison") == token:
            return no_update
        _update_job(job_id, delivered_comparison=token)
        return build_case_comparison_figure(
            job, metric, selected_storage, selected_strategies, selected_cases
        )
    @app.callback(
        Output("capex-breakdown", "figure"), Output("opex-breakdown", "figure"),
        Input("run-job-poll", "n_intervals"), Input("run-job", "data"),
        Input("result-category", "value"), Input("comparison-case-filter", "value"),
        Input("comparison-storage-filter", "value"), Input("comparison-strategy-filter", "value"),
    )
    def update_cost_breakdowns(_, data, category, selected_cases, storage, strategies):
        job = _job_snapshot(data["job_id"]) if data else None
        if not job:
            message = "Run a case to display its capital and operating cost breakdown."
            return empty_comparison_figure(message), empty_comparison_figure(message)
        token = json.dumps([job.get("status"), category, selected_cases, storage, strategies])
        if job.get("delivered_cost_breakdown") == token:
            return no_update, no_update
        _update_job(data["job_id"], delivered_cost_breakdown=token)
        if job.get("status") != "complete":
            message = ("The case calculation failed." if job.get("status") == "failed"
                       else "Calculations are still running.")
            return empty_comparison_figure(message), empty_comparison_figure(message)
        return build_cost_breakdown_figures(job, category, selected_cases, storage, strategies)

    @app.callback(Output("pfd-case", "options"), Output("pfd-case", "value"),
                  Input("run-job-poll", "n_intervals"), Input("run-job", "data"),
                  Input("completed-case", "value"), State("pfd-case", "value"),
                  State("pfd-case", "options"))
    def pfd_case_options(_, data, main_case, current, current_options):
        job = _job_snapshot(data["job_id"]) if data else None
        if not job or job.get("status") != "complete":
            return ([], None) if current_options or current else (no_update, no_update)
        options = [{"label": job.get("case_labels", {}).get(key, key), "value": key}
                   for key in job["results"]]
        preferred = main_case if ctx.triggered_id == "completed-case" else current
        value = preferred if preferred in job["results"] else next(iter(job["results"]), None)
        return (options if options != current_options else no_update,
                value if value != current else no_update)

    @app.callback(Output("pfd-category", "value"), Input("result-category", "value"))
    def mirror_pfd_category(category):
        return category

    @app.callback(Output("pfd-frame", "srcDoc"), Output("pfd-status", "children"),
                  Output("pfd-stream-table", "children"), Output("pfd-render-key", "data"),
                  Input("run-job-poll", "n_intervals"), Input("run-job", "data"),
                  Input("pfd-case", "value"), Input("pfd-category", "value"),
                  State("pfd-render-key", "data"))
    def update_pfd(_, data, selected_case, category, previous):
        job_id = data.get("job_id") if data else None
        job = _job_snapshot(job_id) if job_id else None
        key = selected_case if job and selected_case in job.get("results", {}) else (
            next(iter(job.get("results", {})), None) if job else None)
        actual_category = "perfect" if job and job.get("information_mode") == "perfect_only" else category
        token = [job_id, job.get("status") if job else None, key, actual_category]
        if token == previous:
            return no_update, no_update, no_update, no_update
        summary = None
        if not job:
            message = "Run a case to populate the stream labels."
        elif job.get("status") in {"queued", "running"}:
            message = "The latest run is still calculating. Stream values will appear when it completes."
        elif job.get("status") == "failed":
            message = "The latest run failed; no stream values are available."
        elif key is None:
            message = "No completed cases are available."
        else:
            result = job["results"][key]
            missing_faults = actual_category == "imperfect_with_faults" and getattr(result, "imperfect_with_faults", None) is None
            simulation, _, label = _result_category_parts(result, actual_category)
            if missing_faults or simulation is None:
                message = "This result category was not calculated. Select another category."
            else:
                summary = simulation.metadata.get("pfd_summary")
                if summary is None:
                    message = "Run this case again to calculate its stream averages."
                else:
                    case_label = job.get("case_labels", {}).get(key, key)
                    message = (f"{case_label} | {label} | {summary['hours']:,} hourly samples, including shutdowns. "
                               f"Period: {summary['start']} to {summary['end']}.")
        table = []
        if summary:
            rows = []
            for tag, stream in summary["streams"].items():
                rows.append(html.Tr([html.Td(tag), html.Td(stream["name"]),
                                    html.Td(format_number(stream["flow_kg_h"])),
                                    html.Td(format_number(stream["temperature_c"])),
                                    html.Td(format_number(stream["pressure_bar"])),
                                    html.Td(f"{stream['note']} Temperature: {stream['temperature_basis']}. "
                                            f"Pressure: {stream['pressure_basis']}.")]))
            table = html.Table([html.Thead(html.Tr([html.Th(label) for label in
                               ("Stream", "Material", "Average kg/h", "Temperature °C", "Pressure bar", "Basis")])),
                                html.Tbody(rows)], className="pfd-table")
        # Keep the diagram subtitle short; the full case and period remain above it.
        diagram_message = (f"Latest run | {actual_category.replace('_', ' ')} | "
                           f"{summary['hours']:,} h averages including shutdowns | * assumed; N/A not modelled"
                           if summary else message)
        return render_pfd(summary, diagram_message), message, table, token

    def group_dashboard(node):
        if isinstance(node, dcc.Tab) and node.label == "Single site":
            children = node.children
            def find_id(component_id):
                return next(i for i, child in enumerate(children) if getattr(child, "id", None) == component_id)
            metrics = find_id("metric-cards")
            children[metrics] = _section("Performance metrics", children[metrics], "#176b55")
            start = find_id("load-dispatch")
            end = find_id("reactor-count-graph") + 1
            children[start:end] = [_section("Dispatch & reactor activity", children[start:end], "#315f8c")]
            start = next(i for i, child in enumerate(children) if isinstance(child, html.H3))
            comparison = children[start + 1:]
            comparison[2] = _section("Filter cases", comparison[2], "#b07722")
            children[start:] = [_section("Test-case comparison", comparison, "#b07722", True)]
            children.append(_section("Capital & operating costs", [
                dcc.Graph(id="capex-breakdown", figure=empty_comparison_figure("Run a case to display its capital cost breakdown."),
                          config={"responsive": True, "displaylogo": False}),
                dcc.Graph(id="opex-breakdown", figure=empty_comparison_figure("Run a case to display its annual operating cost breakdown."),
                          config={"responsive": True, "displaylogo": False}),
            ], "#276b72", True, help_text=(
                "One stacked bar per completed case. Capital costs show installed equipment, solar and storage; "
                "operating costs show annual fixed costs and production-dependent costs. Storage includes its "
                "associated conversion equipment, counted once. Uses the selected result category and the "
                "filters in Test-case comparison. Hover over a segment for its amount and the case total."
            )))
            return
        children = getattr(node, "children", [])
        for child in children if isinstance(children, list) else [children]:
            group_dashboard(child)
    group_dashboard(app.layout)

    @app.callback(Output("run-all", "children"),
                  Input({"type": "investigate", "parameter": ALL}, "value"))
    def investigation_button(values):
        count = sum(bool(value) for value in values)
        return f"Investigate {count} selected parameter{'s' if count != 1 else ''}"

    @app.callback(Output("completed-case", "options"), Output("completed-case", "value"),
                  Input("comparison-case-filter", "options"))
    def completed_case_options(options):
        return options, None

    @app.callback(Output("comparison-case-filter", "options"),
                  Output("comparison-case-filter", "value"),
                  Input("run-job-poll", "n_intervals"), State("run-job", "data"),
                  State("comparison-case-filter", "options"))
    def available_test_cases(_, data, current):
        job = _job_snapshot(data["job_id"]) if data else None
        if not job or job.get("status") != "complete":
            return no_update, no_update
        options = [{"label": job.get("case_labels", {}).get(key, key), "value": key}
                   for key in job["results"]]
        if options == current:
            return no_update, no_update
        return options, list(job["results"])

    def tidy_notes(node):
        if not hasattr(node, "children"):
            return
        children = node.children
        items = children if isinstance(children, list) else [children]
        for index, child in enumerate(items):
            if isinstance(child, (html.P, html.Small, html.Span)) and not getattr(child, "id", None):
                content = getattr(child, "children", None)
                parts = content if isinstance(content, list) else [content]
                if parts and all(isinstance(part, str) for part in parts) and len("".join(parts)) > 85:
                    items[index] = _info("".join(parts))
                    continue
            tidy_notes(child)
        node.children = items if isinstance(children, list) else items[0]
    tidy_notes(app.layout)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False)
