"""Small, auditable calculation functions used by the notebook and model.

These functions deliberately do not know about locations, forecasts, operating
strategies, time-series simulation, or whole-plant cases. Each function represents
one equation or one local bookkeeping operation. Canonical units are kW, kWh, kg,
h, K, bar, and USD.
"""

from __future__ import annotations

import math
from typing import Mapping

try:
    from CoolProp.CoolProp import PropsSI
except ImportError:  # pragma: no cover
    PropsSI = None


MOLAR_MASS_KG_PER_MOL = {
    "CH4": 0.0160425,
    "H2": 0.00201588,
    "CO2": 0.0440095,
    "H2O": 0.01801528,
    "CaO": 0.056077,
    "CaCO3": 0.1000869,
}
R_J_PER_MOL_K = 8.314462618
H2_LHV_KWH_PER_KG = 33.33


def methane_stoichiometry(
    methane_kg: float = 1.0,
    molar_masses_kg_per_mol: Mapping[str, float] = MOLAR_MASS_KG_PER_MOL,
) -> dict[str, float]:
    """Masses for ``CO2 + 4 H2 -> CH4 + 2 H2O``.

    Returns kg of each material for the requested kg of methane. One mole of CaO
    and CaCO3 is circulated for every mole of captured CO2.
    """
    if methane_kg < 0:
        raise ValueError("Methane mass cannot be negative")
    methane_moles = methane_kg / molar_masses_kg_per_mol["CH4"]
    return {
        "CH4": methane_kg,
        "H2": 4 * methane_moles * molar_masses_kg_per_mol["H2"],
        "CO2": methane_moles * molar_masses_kg_per_mol["CO2"],
        "H2O_product": 2 * methane_moles * molar_masses_kg_per_mol["H2O"],
        "CaO_circulating": methane_moles * molar_masses_kg_per_mol["CaO"],
        "CaCO3_circulating": methane_moles * molar_masses_kg_per_mol["CaCO3"],
    }


def electrolysis_duty_kwh(hydrogen_kg: float, specific_energy_kwh_per_kg_h2: float) -> float:
    """Electrical duty ``E = m_H2 * xi_electrolyser``."""
    _require_nonnegative(hydrogen_kg=hydrogen_kg, specific_energy=specific_energy_kwh_per_kg_h2)
    return hydrogen_kg * specific_energy_kwh_per_kg_h2


def fan_duty_kwh(co2_kg: float, specific_energy_kwh_per_kg_co2: float) -> float:
    """DAC fan duty ``E = m_CO2 * eta_fan``."""
    _require_nonnegative(co2_kg=co2_kg, specific_energy=specific_energy_kwh_per_kg_co2)
    return co2_kg * specific_energy_kwh_per_kg_co2


def reaction_heat_kwh(material_kg: float, specific_heat_kwh_per_kg: float) -> float:
    """Reaction heat magnitude on a material-mass basis."""
    _require_nonnegative(material_kg=material_kg, specific_heat=specific_heat_kwh_per_kg)
    return material_kg * specific_heat_kwh_per_kg


def gas_compression_work_kwh_per_kg(
    fluid: str,
    inlet_pressure_bar: float,
    outlet_pressure_bar: float,
    inlet_temperature_k: float,
    isentropic_efficiency: float,
    stages: int,
    fallback_cp_j_per_kg_k: float = 846.0,
    fallback_gamma: float = 1.289,
) -> tuple[float, str | None]:
    """Equal pressure-ratio gas stages with full intercooling.

    CoolProp supplies enthalpy and entropy. An ideal-gas estimate is used if the
    requested state cannot be evaluated.
    """
    if inlet_pressure_bar <= 0 or outlet_pressure_bar <= 0:
        raise ValueError("Pressures must be positive")
    if outlet_pressure_bar <= inlet_pressure_bar:
        return 0.0, None
    if stages < 1 or not 0 < isentropic_efficiency <= 1:
        raise ValueError("Stages and isentropic efficiency are invalid")
    pressure_ratio = (outlet_pressure_bar / inlet_pressure_bar) ** (1 / stages)
    total_j_per_kg = 0.0
    pressure_bar = inlet_pressure_bar
    warning = None
    for _ in range(stages):
        p1_pa = pressure_bar * 1e5
        p2_pa = p1_pa * pressure_ratio
        if PropsSI is not None:
            try:
                h1 = PropsSI("H", "T", inlet_temperature_k, "P", p1_pa, fluid)
                s1 = PropsSI("S", "T", inlet_temperature_k, "P", p1_pa, fluid)
                h2_isentropic = PropsSI("H", "P", p2_pa, "S", s1, fluid)
                stage_work = (h2_isentropic - h1) / isentropic_efficiency
            except (ValueError, RuntimeError):
                stage_work = _ideal_gas_compression_stage(
                    inlet_temperature_k, pressure_ratio, isentropic_efficiency,
                    fallback_cp_j_per_kg_k, fallback_gamma,
                )
                warning = f"CoolProp state failed; ideal-gas {fluid} estimate used."
        else:
            stage_work = _ideal_gas_compression_stage(
                inlet_temperature_k, pressure_ratio, isentropic_efficiency,
                fallback_cp_j_per_kg_k, fallback_gamma,
            )
            warning = f"CoolProp unavailable; ideal-gas {fluid} estimate used."
        total_j_per_kg += stage_work
        pressure_bar *= pressure_ratio
    return total_j_per_kg / 3.6e6, warning


def co2_compression_work_kwh_per_kg(
    inlet_pressure_bar: float, outlet_pressure_bar: float, inlet_temperature_k: float,
    isentropic_efficiency: float, stages: int,
    fallback_cp_j_per_kg_k: float = 846.0, fallback_gamma: float = 1.289,
) -> tuple[float, str | None]:
    """Backward-compatible CO2 wrapper around the generic gas compressor."""
    return gas_compression_work_kwh_per_kg(
        "CO2", inlet_pressure_bar, outlet_pressure_bar, inlet_temperature_k,
        isentropic_efficiency, stages, fallback_cp_j_per_kg_k, fallback_gamma,
    )


def gas_expansion_work_kwh_per_kg(
    fluid: str, inlet_pressure_bar: float, outlet_pressure_bar: float,
    inlet_temperature_k: float, isentropic_efficiency: float, stages: int,
    fallback_cp_j_per_kg_k: float, fallback_gamma: float,
) -> tuple[float, float, str | None]:
    """Work and interstage reheat duty for equal pressure-ratio expansion stages.

    Every stage begins at ``inlet_temperature_k``, so the gas is reheated to that
    reference temperature between stages. Returns
    ``(recoverable_work, reheat_duty, warning)`` in kWh/kg; the reheat duty is a
    heat input that must be supplied, not a free assumption. For an ideal gas with
    constant heat capacity the two are exactly equal, because the enthalpy removed
    as work is precisely what the reheater must return.
    """
    if inlet_pressure_bar <= 0 or outlet_pressure_bar <= 0:
        raise ValueError("Pressures must be positive")
    if inlet_pressure_bar <= outlet_pressure_bar:
        return 0.0, 0.0, None
    if stages < 1 or not 0 < isentropic_efficiency <= 1:
        raise ValueError("Stages and isentropic efficiency are invalid")
    pressure_ratio = (outlet_pressure_bar / inlet_pressure_bar) ** (1 / stages)
    total_j_per_kg = 0.0
    total_reheat_j_per_kg = 0.0
    pressure_bar = inlet_pressure_bar
    warning = None
    for _ in range(stages):
        p1_pa = pressure_bar * 1e5
        p2_pa = p1_pa * pressure_ratio
        if PropsSI is not None:
            try:
                h1 = PropsSI("H", "T", inlet_temperature_k, "P", p1_pa, fluid)
                s1 = PropsSI("S", "T", inlet_temperature_k, "P", p1_pa, fluid)
                h2_isentropic = PropsSI("H", "P", p2_pa, "S", s1, fluid)
                stage_work = max(0.0, (h1 - h2_isentropic) * isentropic_efficiency)
                # Reheat the expanded gas back to the reference temperature so the
                # next stage starts from the modelled inlet state.
                h2_actual = h1 - stage_work
                h2_reheated = PropsSI("H", "T", inlet_temperature_k, "P", p2_pa, fluid)
                stage_reheat = max(0.0, h2_reheated - h2_actual)
            except (ValueError, RuntimeError):
                stage_work = _ideal_gas_expansion_stage(
                    inlet_temperature_k, pressure_ratio, isentropic_efficiency,
                    fallback_cp_j_per_kg_k, fallback_gamma,
                )
                stage_reheat = stage_work
                warning = f"CoolProp state failed; ideal-gas {fluid} expansion estimate used."
        else:
            stage_work = _ideal_gas_expansion_stage(
                inlet_temperature_k, pressure_ratio, isentropic_efficiency,
                fallback_cp_j_per_kg_k, fallback_gamma,
            )
            stage_reheat = stage_work
            warning = f"CoolProp unavailable; ideal-gas {fluid} expansion estimate used."
        total_j_per_kg += stage_work
        total_reheat_j_per_kg += stage_reheat
        pressure_bar *= pressure_ratio
    return total_j_per_kg / 3.6e6, total_reheat_j_per_kg / 3.6e6, warning


def _ideal_gas_compression_stage(
    inlet_temperature_k: float, pressure_ratio: float, isentropic_efficiency: float,
    cp_j_per_kg_k: float = 846.0, gamma: float = 1.289,
) -> float:
    isentropic_work = cp_j_per_kg_k * inlet_temperature_k * (
        pressure_ratio ** ((gamma - 1) / gamma) - 1
    )
    return isentropic_work / isentropic_efficiency


def _ideal_gas_expansion_stage(
    inlet_temperature_k: float, pressure_ratio: float, isentropic_efficiency: float,
    cp_j_per_kg_k: float, gamma: float,
) -> float:
    if not 0 < pressure_ratio < 1 or cp_j_per_kg_k <= 0 or gamma <= 1:
        raise ValueError("Ideal-gas expansion inputs are invalid")
    isentropic_work = cp_j_per_kg_k * inlet_temperature_k * (
        1 - pressure_ratio ** ((gamma - 1) / gamma)
    )
    return isentropic_work * isentropic_efficiency


def gas_sensible_heat_kwh_per_kg(
    fluid: str, inlet_temperature_k: float, outlet_temperature_k: float, pressure_bar: float,
    fallback_cp_j_per_kg_k: float | None = None,
) -> tuple[float, str | None]:
    """Sensible enthalpy rise of H2 or CO2 at constant pressure."""
    if outlet_temperature_k <= inlet_temperature_k:
        return 0.0, None
    if pressure_bar <= 0:
        raise ValueError("Pressure must be positive")
    if PropsSI is not None:
        try:
            h1 = PropsSI("H", "T", inlet_temperature_k, "P", pressure_bar * 1e5, fluid)
            h2 = PropsSI("H", "T", outlet_temperature_k, "P", pressure_bar * 1e5, fluid)
            return max(0.0, (h2 - h1) / 3.6e6), None
        except (ValueError, RuntimeError):
            pass
    fallback_cp = {"H2": 14_300.0, "CO2": 900.0}
    if fluid not in fallback_cp:
        raise ValueError(f"No fallback heat capacity for {fluid}")
    cp = fallback_cp[fluid] if fallback_cp_j_per_kg_k is None else fallback_cp_j_per_kg_k
    duty = cp * (outlet_temperature_k - inlet_temperature_k) / 3.6e6
    return duty, f"CoolProp {fluid} state failed; constant-Cp estimate used."


def air_sensible_heat_kwh_per_kg(
    inlet_temperature_k: float,
    outlet_temperature_k: float,
    pressure_bar: float,
    fallback_cp_kwh_per_kg_k: float,
) -> tuple[float, str | None]:
    """Air sensible heat from CoolProp with an explicit constant-Cp fallback."""
    if outlet_temperature_k <= inlet_temperature_k:
        return 0.0, None
    if pressure_bar <= 0 or fallback_cp_kwh_per_kg_k <= 0:
        raise ValueError("Air pressure and fallback heat capacity must be positive")
    if PropsSI is not None:
        try:
            pressure_pa = pressure_bar * 1e5
            h1 = PropsSI("H", "T", inlet_temperature_k, "P", pressure_pa, "Air")
            h2 = PropsSI("H", "T", outlet_temperature_k, "P", pressure_pa, "Air")
            return max(0.0, (h2 - h1) / 3.6e6), None
        except (ValueError, RuntimeError):
            pass
    duty = fallback_cp_kwh_per_kg_k * (outlet_temperature_k - inlet_temperature_k)
    return duty, "CoolProp Air state failed; constant-Cp estimate used."


def solids_sensible_loss_kwh(
    cao_kg: float,
    caco3_kg: float,
    cao_cp_kwh_per_kg_k: float,
    caco3_cp_kwh_per_kg_k: float,
    cold_temperature_k: float,
    hot_temperature_k: float,
    recovery_efficiency: float,
) -> dict[str, float]:
    """Gross and unrecovered sensible heat for the circulating solids."""
    if not 0 <= recovery_efficiency <= 1:
        raise ValueError("Recovery efficiency must lie between zero and one")
    delta_t = max(0.0, hot_temperature_k - cold_temperature_k)
    gross = (cao_kg * cao_cp_kwh_per_kg_k + caco3_kg * caco3_cp_kwh_per_kg_k) * delta_t
    return {
        "gross_solids_sensible_heat": gross,
        "recovered_solids_sensible_heat": gross * recovery_efficiency,
        "unrecovered_solids_sensible_heat": gross * (1 - recovery_efficiency),
    }


def dry_air_mass_for_co2_kg(
    co2_kg: float,
    co2_mole_fraction: float = 400e-6,
    air_molar_mass_kg_per_mol: float = 0.028965,
    co2_molar_mass_kg_per_mol: float = MOLAR_MASS_KG_PER_MOL["CO2"],
    capture_efficiency: float = 1.0,
) -> float:
    """Dry-air mass that must be processed to capture the requested CO2.

    ``capture_efficiency`` is the single-pass fraction of the incoming CO2 that
    the contactor actually removes, so the processed air mass scales as
    ``1 / capture_efficiency``. The default of one reproduces the complete-capture
    basis.
    """
    if co2_mole_fraction <= 0:
        raise ValueError("CO2 mole fraction must be positive")
    if not 0 < capture_efficiency <= 1:
        raise ValueError("Capture efficiency must lie above zero and at or below one")
    co2_moles = co2_kg / co2_molar_mass_kg_per_mol
    return co2_moles / (co2_mole_fraction * capture_efficiency) * air_molar_mass_kg_per_mol


def air_heating_duty_kwh(
    air_kg: float,
    cp_kwh_per_kg_k: float,
    inlet_temperature_k: float,
    outlet_temperature_k: float,
    feed_exhaust_effectiveness: float,
) -> dict[str, float]:
    """Gross air heating and the residual after feed/exhaust exchange."""
    if not 0 <= feed_exhaust_effectiveness <= 1:
        raise ValueError("Heat-exchanger effectiveness must lie between zero and one")
    gross = air_kg * cp_kwh_per_kg_k * max(0.0, outlet_temperature_k - inlet_temperature_k)
    return {
        "gross_air_heating": gross,
        "recovered_air_heating": gross * feed_exhaust_effectiveness,
        "residual_air_heating": gross * (1 - feed_exhaust_effectiveness),
    }


def sum_specific_electricity_kwh_per_kg(
    breakdown: Mapping[str, float], excluded_keys: tuple[str, ...] = ()
) -> float:
    """Sum the consuming terms of an auditable energy breakdown."""
    return sum(value for key, value in breakdown.items() if key not in excluded_keys)


def calcination_equilibrium_pressure_bar(
    temperature_k: float,
    delta_h_j_per_mol: float,
    delta_s_j_per_mol_k: float,
    gas_constant_j_per_mol_k: float = R_J_PER_MOL_K,
) -> float:
    """van't Hoff estimate ``p_eq/bar = exp(DeltaS/R - DeltaH/RT)``."""
    if temperature_k <= 0:
        raise ValueError("Temperature must be positive")
    return math.exp(
        delta_s_j_per_mol_k / gas_constant_j_per_mol_k
        - delta_h_j_per_mol / (gas_constant_j_per_mol_k * temperature_k)
    )


def maximum_carbonation_temperature_k(
    co2_partial_pressure_bar: float,
    delta_h_j_per_mol: float,
    delta_s_j_per_mol_k: float,
    gas_constant_j_per_mol_k: float = R_J_PER_MOL_K,
) -> float:
    """Highest temperature at which CaO can carbonate in the supplied gas.

    Inverts ``calcination_equilibrium_pressure_bar`` for the temperature where the
    equilibrium CO2 pressure equals the partial pressure actually offered, i.e.
    ``T = DeltaH / (DeltaS - R ln p)``. Carbonation requires a temperature below
    this value; above it CaCO3 decomposes instead of forming.
    """
    if co2_partial_pressure_bar <= 0:
        raise ValueError("CO2 partial pressure must be positive")
    denominator = (delta_s_j_per_mol_k
                   - gas_constant_j_per_mol_k * math.log(co2_partial_pressure_bar))
    if denominator <= 0:
        return math.inf
    return delta_h_j_per_mol / denominator


def thermal_inventory_kwh_per_k(
    mass_flow_kg_h: float, cycle_time_h: float, cp_kwh_per_kg_k: float
) -> tuple[float, float]:
    """Return inventory kg and lumped heat capacity kWh/K."""
    _require_nonnegative(mass_flow=mass_flow_kg_h, cycle_time=cycle_time_h, cp=cp_kwh_per_kg_k)
    inventory = mass_flow_kg_h * cycle_time_h
    return inventory, inventory * cp_kwh_per_kg_k


def lumped_cooling_step_k(
    temperature_k: float,
    ambient_temperature_k: float,
    ua_kw_per_k: float,
    heat_capacity_kwh_per_k: float,
    timestep_h: float = 1.0,
) -> float:
    """One exact time step of ``C dT/dt = -UA(T-Tamb)``."""
    if heat_capacity_kwh_per_k <= 0 or timestep_h < 0 or ua_kw_per_k < 0:
        raise ValueError("Thermal capacity, UA, or timestep is invalid")
    retention = math.exp(-ua_kw_per_k * timestep_h / heat_capacity_kwh_per_k)
    return ambient_temperature_k + (temperature_k - ambient_temperature_k) * retention


def battery_charge_step(
    soc_kwh: float,
    available_bus_kwh: float,
    capacity_kwh: float,
    max_charge_kw: float,
    timestep_h: float = 1.0,
) -> tuple[float, float]:
    """Return ``(new_SOC, bus_energy_accepted)`` for an ideal battery charge step."""
    room = max(0.0, capacity_kwh - soc_kwh)
    accepted = min(max(0.0, available_bus_kwh), max_charge_kw * timestep_h, room)
    return soc_kwh + accepted, accepted


def battery_discharge_step(
    soc_kwh: float,
    bus_demand_kwh: float,
    max_discharge_kw: float,
    timestep_h: float = 1.0,
) -> tuple[float, float]:
    """Return ``(new_SOC, bus_energy_delivered)`` for an ideal battery discharge step."""
    delivered = min(max(0.0, bus_demand_kwh), max_discharge_kw * timestep_h,
                    max(0.0, soc_kwh))
    return soc_kwh - delivered, delivered


def hydrogen_mass_from_lhv_kwh(stored_lhv_kwh: float) -> float:
    """Convert stored hydrogen LHV energy to kg H2."""
    return stored_lhv_kwh / H2_LHV_KWH_PER_KG


def capital_recovery_factor(real_discount_rate: float, project_life_years: int) -> float:
    """Annual capital recovery factor for a constant real discount rate."""
    if project_life_years <= 0:
        raise ValueError("Project life must be positive")
    if real_discount_rate == 0:
        return 1 / project_life_years
    rate = real_discount_rate
    return rate * (1 + rate) ** project_life_years / ((1 + rate) ** project_life_years - 1)


def scale_equipment_cost(
    reference_cost: float,
    capacity: float,
    reference_capacity: float,
    scaling_exponent: float,
) -> float:
    """Six-tenths-style scaling ``C=C_ref(S/S_ref)^n``."""
    if capacity <= 0:
        return 0.0
    if reference_cost < 0 or reference_capacity <= 0 or scaling_exponent <= 0:
        raise ValueError("Cost-scaling inputs are invalid")
    return reference_cost * (capacity / reference_capacity) ** scaling_exponent


def _require_nonnegative(**values: float) -> None:
    invalid = [name for name, value in values.items() if value < 0]
    if invalid:
        raise ValueError(f"Values must be nonnegative: {', '.join(invalid)}")


__all__ = [
    "H2_LHV_KWH_PER_KG",
    "MOLAR_MASS_KG_PER_MOL",
    "air_sensible_heat_kwh_per_kg",
    "air_heating_duty_kwh",
    "battery_charge_step",
    "battery_discharge_step",
    "calcination_equilibrium_pressure_bar",
    "capital_recovery_factor",
    "co2_compression_work_kwh_per_kg", "gas_compression_work_kwh_per_kg",
    "gas_expansion_work_kwh_per_kg",
    "dry_air_mass_for_co2_kg",
    "electrolysis_duty_kwh",
    "fan_duty_kwh",
    "gas_sensible_heat_kwh_per_kg",
    "hydrogen_mass_from_lhv_kwh",
    "lumped_cooling_step_k",
    "maximum_carbonation_temperature_k",
    "methane_stoichiometry",
    "reaction_heat_kwh",
    "scale_equipment_cost",
    "solids_sensible_loss_kwh",
    "sum_specific_electricity_kwh_per_kg",
    "thermal_inventory_kwh_per_k",
]
