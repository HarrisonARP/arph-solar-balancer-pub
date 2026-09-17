import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import functions as calculations
import model


class PlantEnergyTests(unittest.TestCase):
    def test_stoichiometry_is_mass_consistent(self):
        values = calculations.methane_stoichiometry()
        self.assertAlmostEqual(values["CO2"], 2.7433, places=3)
        self.assertAlmostEqual(values["H2"], 0.5026, places=3)
        self.assertAlmostEqual(values["CO2"] + values["H2"], 1 + values["H2O_product"], places=5)

    def test_energy_breakdown_sums_and_is_positive(self):
        result = model.calculate_plant_energy()
        included = sum(v for k, v in result.breakdown_kwh_per_kg_ch4.items()
                       if k not in {"sabatier_heat_recovered_to_feed", "sabatier_heat_discarded"})
        self.assertAlmostEqual(result.e_req_kwh_per_kg_ch4, included)
        self.assertAlmostEqual(
            result.breakdown_kwh_per_kg_ch4["sabatier_heat_recovered_to_feed"]
            + result.breakdown_kwh_per_kg_ch4["sabatier_heat_discarded"],
            model.PlantParameters().sabatier_heat_kwh_per_kg_ch4,
        )
        self.assertGreater(result.e_req_kwh_per_kg_ch4, 30)
        self.assertLess(result.e_req_kwh_per_kg_ch4, 100)

    def test_maximum_carbonation_temperature_inverts_the_equilibrium_pressure(self):
        plant = model.PlantParameters()
        pressure = plant.air_co2_mole_fraction * plant.air_pressure_bar
        limit = calculations.maximum_carbonation_temperature_k(
            pressure, plant.calcination_delta_h_j_per_mol,
            plant.calcination_delta_s_j_per_mol_k, plant.gas_constant_j_per_mol_k,
        )
        # The inverse must land exactly back on the forward van't Hoff relation.
        self.assertAlmostEqual(
            calculations.calcination_equilibrium_pressure_bar(
                limit, plant.calcination_delta_h_j_per_mol,
                plant.calcination_delta_s_j_per_mol_k, plant.gas_constant_j_per_mol_k,
            ),
            pressure, places=12,
        )
        # A richer gas tolerates a hotter carbonator.
        self.assertGreater(
            calculations.maximum_carbonation_temperature_k(
                0.15, plant.calcination_delta_h_j_per_mol,
                plant.calcination_delta_s_j_per_mol_k, plant.gas_constant_j_per_mol_k,
            ),
            limit,
        )
        with self.assertRaises(ValueError):
            calculations.maximum_carbonation_temperature_k(
                0.0, plant.calcination_delta_h_j_per_mol,
                plant.calcination_delta_s_j_per_mol_k,
            )

    def test_carbonator_feasibility_is_reported_and_warned(self):
        feasible = model.calculate_plant_energy()
        self.assertGreater(feasible.carbonation_driving_force_ratio, 1.0)
        self.assertLess(
            model.PlantParameters().carbonator_temperature_k,
            feasible.maximum_carbonator_temperature_k,
        )
        self.assertAlmostEqual(
            feasible.carbonation_driving_force_ratio,
            feasible.air_co2_partial_pressure_bar
            / feasible.carbonation_equilibrium_pressure_bar,
        )
        self.assertFalse([w for w in feasible.warnings if "cannot capture CO2" in w])

        # The previous 923.15 K default is 49x oversaturated and must be flagged.
        infeasible = model.calculate_plant_energy(
            replace(model.PlantParameters(), carbonator_temperature_k=923.15)
        )
        self.assertLess(infeasible.carbonation_driving_force_ratio, 1.0)
        self.assertTrue([w for w in infeasible.warnings if "cannot capture CO2" in w])

        # Lowering the carbonator cuts the dominant air duty and raises the much
        # smaller solids duty, because the solids now span a wider loop.
        cool = feasible.breakdown_kwh_per_kg_ch4
        hot = infeasible.breakdown_kwh_per_kg_ch4
        self.assertLess(cool["carbonator_air_electric_heat"],
                        hot["carbonator_air_electric_heat"])
        self.assertGreater(cool["unrecovered_solids_sensible_heat"],
                           hot["unrecovered_solids_sensible_heat"])
        self.assertLess(feasible.e_req_kwh_per_kg_ch4, infeasible.e_req_kwh_per_kg_ch4)

    def test_expander_reheat_is_charged_as_electrical_heat(self):
        plant = model.PlantParameters()
        result = model.calculate_plant_energy(plant)
        work = result.gas_storage_work_kwh_per_kg
        for gas in ("hydrogen", "co2", "methane"):
            self.assertGreater(work[f"{gas}_expansion_reheat"], 0.0)
        # The reheat is a positive electric term inside E_req, so the product
        # methane storage loop must be a net energy sink rather than a source.
        self.assertGreater(result.breakdown_kwh_per_kg_ch4["methane_storage_expansion_reheat"], 0.0)
        net_methane_storage = (
            result.breakdown_kwh_per_kg_ch4["methane_storage_compression"]
            + result.breakdown_kwh_per_kg_ch4["methane_storage_expansion_recovered"]
            + result.breakdown_kwh_per_kg_ch4["methane_storage_expansion_reheat"]
        )
        self.assertGreater(net_methane_storage, 0.0)
        # An ideal gas with constant Cp returns exactly the work as reheat.
        ideal_work, ideal_reheat, warning = calculations.gas_expansion_work_kwh_per_kg(
            "NotAFluid", 300.0, 40.0, 313.15, 0.8, 3, 2220.0, 1.31,
        )
        self.assertIsNotNone(warning)
        self.assertGreater(ideal_work, 0.0)
        self.assertAlmostEqual(ideal_reheat, ideal_work, places=12)
        # No pressure difference means no machine, so neither term exists.
        self.assertEqual(
            calculations.gas_expansion_work_kwh_per_kg(
                "CO2", 40.0, 40.0, 313.15, 0.8, 3, 846.0, 1.289)[:2],
            (0.0, 0.0),
        )

    def test_net_letdown_energy_is_a_first_law_invariant(self):
        """Work less reheat must equal -dh over the isothermal letdown.

        A steady-state balance over the whole machine, store state to reference
        temperature at the connection pressure, gives Q - W = h_out - h_in. The net
        bus cost of getting gas out of storage is therefore fixed by the end states
        alone: it cannot be changed by adding stages or by a better expander, and a
        plain throttle valve would incur the same net. Every extra kilowatt-hour the
        expander recovers costs exactly one more kilowatt-hour of reheat.
        """
        base = model.PlantParameters()
        expected = {}
        for stages in (1, 3, 6):
            for efficiency in (0.6, 0.8, 1.0):
                plant = replace(
                    base, storage_machine_stages=stages,
                    storage_expander_isentropic_efficiency=efficiency,
                )
                work = model.calculate_plant_energy(plant).gas_storage_work_kwh_per_kg
                for gas in ("hydrogen", "co2", "methane"):
                    net = work[f"{gas}_expansion"] - work[f"{gas}_expansion_reheat"]
                    expected.setdefault(gas, net)
                    self.assertAlmostEqual(
                        net, expected[gas], places=9,
                        msg=f"{gas} net letdown moved at {stages} stages, eta={efficiency}",
                    )
                # More stages and a better expander must both recover more work.
                self.assertGreater(work["co2_expansion"], 0.0)
        # Only hydrogen, which sits above its Joule-Thomson inversion temperature,
        # yields net bus energy on letdown; the other two are net loads.
        self.assertGreater(expected["hydrogen"], 0.0)
        self.assertLess(expected["co2"], 0.0)
        self.assertLess(expected["methane"], 0.0)

    def test_hydrogen_store_rejects_reheat_that_cancels_discharge(self):
        with self.assertRaises(ValueError):
            model.StorageParameters(
                method="hydrogen", hydrogen_expansion_reheat_kwh_per_kg=99.0,
            )

    def test_dac_capture_efficiency_scales_processed_air_and_air_heating(self):
        complete = calculations.dry_air_mass_for_co2_kg(2.7433, capture_efficiency=1.0)
        partial = calculations.dry_air_mass_for_co2_kg(2.7433, capture_efficiency=0.75)
        self.assertAlmostEqual(partial, complete / 0.75, places=6)
        for invalid in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                calculations.dry_air_mass_for_co2_kg(2.7433, capture_efficiency=invalid)
        with self.assertRaises(ValueError):
            model.PlantParameters(dac_capture_efficiency=0.0)
        with self.assertRaises(ValueError):
            model.PlantParameters(dac_capture_efficiency=1.2)

        base = model.PlantParameters(dac_capture_efficiency=1.0)
        partial_plant = replace(base, dac_capture_efficiency=0.75)
        full_air = model.calculate_plant_energy(base).breakdown_kwh_per_kg_ch4[
            "carbonator_air_electric_heat"]
        partial_air = model.calculate_plant_energy(partial_plant).breakdown_kwh_per_kg_ch4[
            "carbonator_air_electric_heat"]
        self.assertAlmostEqual(partial_air, full_air / 0.75, places=6)
        # Only the air duty is affected; no other breakdown term moves.
        full = model.calculate_plant_energy(base).breakdown_kwh_per_kg_ch4
        part = model.calculate_plant_energy(partial_plant).breakdown_kwh_per_kg_ch4
        for key in full:
            if key != "carbonator_air_electric_heat":
                self.assertAlmostEqual(full[key], part[key], places=9, msg=key)

    def test_calculated_equilibrium_pressure_is_compressor_inlet(self):
        base = model.PlantParameters(co2_outlet_pressure_bar=10)
        high = replace(base, co2_outlet_pressure_bar=40)
        base_result = model.calculate_plant_energy(base)
        high_result = model.calculate_plant_energy(high)
        high_work = calculations.co2_compression_work_kwh_per_kg(
            high_result.calcination_equilibrium_pressure_bar, high.co2_outlet_pressure_bar,
            high.intercool_temperature_k, high.compressor_isentropic_efficiency,
            high.compressor_stages,
        )[0]
        base_work = calculations.co2_compression_work_kwh_per_kg(
            base_result.calcination_equilibrium_pressure_bar, base.co2_outlet_pressure_bar,
            base.intercool_temperature_k, base.compressor_isentropic_efficiency,
            base.compressor_stages,
        )[0]
        self.assertAlmostEqual(base_result.compressor_kwh_per_kg_co2, base_work)
        self.assertAlmostEqual(high_result.compressor_kwh_per_kg_co2, high_work)
        self.assertGreater(high_work, base_work)
        self.assertGreater(
            calculations.calcination_equilibrium_pressure_bar(
                1200, base.calcination_delta_h_j_per_mol, base.calcination_delta_s_j_per_mol_k
            ),
            calculations.calcination_equilibrium_pressure_bar(
                1000, base.calcination_delta_h_j_per_mol, base.calcination_delta_s_j_per_mol_k
            ),
        )

    def test_default_gas_storage_pressures_produce_compressor_and_expander_work(self):
        plant = model.PlantParameters()
        result = model.calculate_plant_energy(plant)
        self.assertEqual(plant.hydrogen_storage_pressure_bar, 300.0)
        self.assertEqual(plant.co2_storage_pressure_bar, 150.0)
        self.assertEqual(plant.methane_storage_pressure_bar, 300.0)
        for gas in ("hydrogen", "co2", "methane"):
            self.assertGreater(result.gas_storage_work_kwh_per_kg[f"{gas}_compression"], 0.0)
            self.assertGreater(result.gas_storage_work_kwh_per_kg[f"{gas}_expansion"], 0.0)

    def test_basic_functions_do_not_expose_case_orchestration(self):
        self.assertFalse(hasattr(calculations, "run_case"))
        self.assertFalse(hasattr(calculations, "simulate_dispatch"))

    def test_ideal_battery_steps_respect_capacity(self):
        charged_soc, accepted = calculations.battery_charge_step(9, 10, 10, 10)
        self.assertAlmostEqual(charged_soc, 10)
        self.assertAlmostEqual(accepted, 1)
        discharged_soc, delivered = calculations.battery_discharge_step(10, 10, 10)
        self.assertAlmostEqual(discharged_soc, 0)
        self.assertAlmostEqual(delivered, 10)


class WeatherTests(unittest.TestCase):
    def test_epoch_millisecond_api_timestamps_are_normalized(self):
        values = pd.Series(["1293840000000", "1293843600000"])
        parsed = model._parse_api_timestamps(values)
        self.assertEqual(parsed[0], pd.Timestamp("2011-01-01 00:00:00", tz="UTC"))
        self.assertEqual(parsed[1] - parsed[0], pd.Timedelta(hours=1))

    def test_annual_yield_falls_monotonically_towards_the_poles(self):
        latitudes = (0.0, 20.0, 37.4, 50.0, 55.9, 69.7)
        means = [
            float(model.make_synthetic_weather(
                "2010-01-01", years=1, latitude_deg=lat
            )["capacity_factor"].mean())
            for lat in latitudes
        ]
        for lower, higher in zip(means, means[1:]):
            self.assertGreater(lower, higher)
        # The gradient has to be worth something, not a rounding artefact.
        self.assertGreater(means[0] / means[-1], 1.5)

    def test_southern_hemisphere_seasons_are_mirrored_and_poles_go_dark(self):
        north = model.make_synthetic_weather("2010-01-01", years=1, latitude_deg=45.0)
        south = model.make_synthetic_weather("2010-01-01", years=1, latitude_deg=-45.0)
        for frame, summer, winter in ((north, 6, 12), (south, 12, 6)):
            monthly = frame["capacity_factor"].groupby(frame.index.month).mean()
            self.assertGreater(monthly[summer], 2 * monthly[winter])
        self.assertAlmostEqual(
            float(north["capacity_factor"].mean()),
            float(south["capacity_factor"].mean()),
            places=2,
        )
        arctic = model.make_synthetic_weather("2010-01-01", years=1, latitude_deg=78.0)
        december = arctic["capacity_factor"][arctic.index.month == 12]
        self.assertEqual(float(december.max()), 0.0)

    def test_cached_only_mode_never_reaches_the_api(self):
        config = model.WeatherConfig(lat=51.5, lon=-0.1, latest_year=2019,
                                     training_years=0, evaluation_years=1)
        frame = model.make_synthetic_weather(start="2019-01-01", years=1)
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp) / model._weather_cache_key(config)
            site.mkdir(parents=True)
            frame.rename_axis("timestamp").to_csv(site / "2019.csv")
            with patch.object(model, "_request_ninja_year") as request:
                loaded, _ = model.fetch_solar_profile(
                    config, cache_dir=temp, allow_api=False,
                )
            request.assert_not_called()
            self.assertEqual(len(loaded), len(frame))

            # A year outside the cache must fail rather than quietly download.
            missing = model.WeatherConfig(lat=51.5, lon=-0.1, latest_year=2020,
                                          training_years=0, evaluation_years=1)
            with patch.object(model, "_request_ninja_year") as request:
                with self.assertRaises(model.WeatherDataError):
                    model.fetch_solar_profile(missing, cache_dir=temp, allow_api=False)
            request.assert_not_called()

    def test_cached_weather_sites_recovers_coordinates_from_metadata(self):
        config = model.WeatherConfig(lat=37.3891, lon=-5.9845, latest_year=2019)
        frame = model.make_synthetic_weather(start="2019-01-01", years=1)
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp) / model._weather_cache_key(config)
            site.mkdir(parents=True)
            frame.rename_axis("timestamp").to_csv(site / "2019.csv")
            (site / "2019.metadata.json").write_text(
                json.dumps({"params": {"lat": "37.3891", "lon": "-5.9845",
                                       "tilt": "35.0", "azim": "180.0"}}),
                encoding="utf-8",
            )
            found = model.cached_weather_sites(temp)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["lat"], 37.3891)
        self.assertAlmostEqual(found[0]["lon"], -5.9845)
        self.assertEqual(found[0]["years"], [2019])

    def test_limiting_subsystem_prefers_faults_then_resource_then_capacity(self):
        index = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
        hourly = pd.DataFrame({
            "curtailed_kwh": [0.0, 0.0, 5.0, 5.0],
            "methane_shortfall_kg": [0.0, 2.0, 0.0, 3.0],
            "dac_capacity_fraction": [1.0, 1.0, 1.0, 0.5],
            "hydrogen_capacity_fraction": [1.0, 1.0, 1.0, 0.9],
            "sabatier_capacity_fraction": [1.0, 1.0, 1.0, 1.0],
            "battery_capacity_fraction": [1.0, 1.0, 1.0, 1.0],
        }, index=index)
        labels = list(model.limiting_subsystem(hourly))
        # Nothing wrong; target missed; surplus curtailed; and a fault outranks both,
        # with the most derated subsystem winning when several are active.
        self.assertEqual(labels, ["At target", "Solar resource", "Plant capacity",
                                  "DAC fault"])

        # A day carrying any real limitation is reported by it, not by quiet hours.
        daily = model.daily_limiting_subsystem(hourly)
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily.iloc[0], "DAC fault")

        quiet = hourly.assign(curtailed_kwh=0.0, methane_shortfall_kg=0.0,
                              dac_capacity_fraction=1.0, hydrogen_capacity_fraction=1.0)
        self.assertEqual(model.daily_limiting_subsystem(quiet).iloc[0], "At target")

    def test_no_storage_baseline_runs_flat_out_and_curtails_more(self):
        weather = model.make_synthetic_weather("2015-01-01", years=2, latitude_deg=51.5)
        training, actual = model.split_weather_period(
            weather, training_years=1, evaluation_years=1)
        forecast = model.build_climatology_forecast(training, actual)
        case = model.run_case(
            actual, forecast, include_baseline=True,
            strategy=model.StrategyConfig(parallel_reactor_count=4),
        )
        self.assertIsNotNone(case.baseline)
        self.assertIsNotNone(case.economics_baseline)

        # The baseline installs no storage at all, so it must waste more sunlight
        # and produce less methane than the scheduled plant it is compared against.
        metadata = case.baseline.metadata
        self.assertEqual(metadata["short_storage"]["capacity_kwh"], 0.0)
        self.assertEqual(metadata["long_storage"]["capacity_kwh"], 0.0)
        self.assertGreater(case.baseline.metrics["curtailment_fraction"],
                           case.perfect.metrics["curtailment_fraction"])
        self.assertLess(case.economics_baseline["average_annual_methane_kg"],
                        case.economics_perfect["average_annual_methane_kg"])
        # Reported as baseline over realised, so the do-nothing reference lands below
        # one rather than above it. Realised is the faulted case when there is one,
        # then the imperfect forecast, and only then perfect information.
        realised = (case.economics_imperfect_with_faults or case.economics_imperfect
                    or case.economics_perfect)
        self.assertLess(case.economics_baseline["baseline_methane_ratio"], 1.0)
        self.assertAlmostEqual(
            case.economics_baseline["baseline_methane_ratio"],
            case.economics_baseline["average_annual_methane_kg"]
            / realised["average_annual_methane_kg"], places=9)

        # Omitting it must leave the case untouched, so existing runs stay cheap.
        without = model.run_case(
            actual, forecast, strategy=model.StrategyConfig(parallel_reactor_count=4),
        )
        self.assertIsNone(without.baseline)
        self.assertIsNone(without.economics_baseline)

    def test_validation_rejects_missing_complete_year_hour(self):
        frame = model.make_synthetic_weather(start="2019-01-01", years=1).iloc[:-1]
        with self.assertRaises(model.WeatherDataError):
            model.validate_hourly_profile(frame, expected_year=2019)

    def test_climatology_interpolates_leap_day(self):
        training = model.make_synthetic_weather(start="2010-01-01", years=5)
        evaluation = model.make_synthetic_weather(start="2016-01-01", years=1)
        forecast = model.build_climatology_forecast(training, evaluation)
        leap = pd.Timestamp("2016-02-29 12:00", tz="UTC")
        self.assertTrue(math.isfinite(forecast.loc[leap, "capacity_factor"]))
        self.assertEqual(len(forecast), 8784)

    def test_cached_year_load_needs_no_token(self):
        config = model.WeatherConfig(lat=51.5, lon=-0.1, latest_year=2019,
                                     training_years=0, evaluation_years=1)
        frame = model.make_synthetic_weather(start="2019-01-01", years=1)
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp) / model._weather_cache_key(config)
            site.mkdir(parents=True)
            frame.rename_axis("timestamp").to_csv(site / "2019.csv")
            (site / "2019.metadata.json").write_text("{}", encoding="utf-8")
            messages = []
            with patch.object(model, "record_request") as record_request:
                loaded, metadata = model.fetch_solar_profile(
                    config, cache_dir=temp, progress=messages.append
                )
                record_request.assert_not_called()
        self.assertEqual(len(loaded), 8760)
        self.assertEqual(metadata["years"], [2019])
        self.assertTrue(any("2019: loading cached data" in message for message in messages))


class FaultGenerationTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2020-01-01", periods=24 * 365, freq="h", tz="UTC")

    def test_distribution_validation_and_zero_disable(self):
        with self.assertRaises(ValueError):
            model.FaultDistribution(mean_months_between_faults=1.5)
        with self.assertRaises(ValueError):
            model.FaultDistribution(mean_duration_h=0)
        with self.assertRaises(ValueError):
            model.FaultDistribution(mean_capacity_fraction=1.1)
        disabled = model.FaultDistribution(mean_months_between_faults=0)
        self.assertEqual(model.generate_battery_faults(self.index, disabled, seed=4), ())

    def test_seeded_streams_are_reproducible_and_independent(self):
        distribution = model.FaultDistribution(mean_months_between_faults=1)
        first = model.generate_battery_faults(self.index, distribution, seed=42)
        second = model.generate_battery_faults(self.index, distribution, seed=42)
        hydrogen = model.generate_hydrogen_faults(self.index, distribution, seed=42)
        self.assertEqual(first, second)
        self.assertNotEqual(first, hydrogen)
        scenario = model.FaultScenario(
            battery=distribution,
            hydrogen=model.FaultDistribution(mean_months_between_faults=0),
            sabatier=model.FaultDistribution(mean_months_between_faults=0),
            dac=model.FaultDistribution(mean_months_between_faults=0),
            seed=42,
        )
        self.assertEqual(first, model.generate_fault_events(self.index, scenario))

    def test_overlaps_use_lowest_capacity_only_during_overlap(self):
        start = self.index[10]
        events = [
            model.FaultEvent("sabatier", start, start + pd.Timedelta(hours=5), 0.8),
            model.FaultEvent("sabatier", start + pd.Timedelta(hours=3),
                             start + pd.Timedelta(hours=7), 0.5),
        ]
        self.assertEqual(model._fault_fraction_at(start + pd.Timedelta(hours=2), "sabatier", events), 0.8)
        self.assertEqual(model._fault_fraction_at(start + pd.Timedelta(hours=4), "sabatier", events), 0.5)
        self.assertEqual(model._fault_fraction_at(start + pd.Timedelta(hours=6), "sabatier", events), 0.5)
        self.assertEqual(model._fault_fraction_at(start + pd.Timedelta(hours=7), "sabatier", events), 1.0)

    def test_legacy_fault_aliases_are_rejected(self):
        for component in ("electrolyser", "short_storage", "long_storage"):
            with self.subTest(component=component), self.assertRaises(ValueError):
                model.FaultEvent(component, self.index[0], self.index[1], 0.25)
        with self.assertRaises(TypeError):
            model.FaultEvent(
                "hydrogen", self.index[0], self.index[1], availability=0.25,
            )

    def test_arbitrary_battery_capacity_faults_conserve_energy_and_recover(self):
        for fraction in (0.0, 0.25, 0.50, 0.80, 0.95):
            with self.subTest(capacity_fraction=fraction):
                store = model._StoreState(
                    model.StorageParameters(capacity_kwh=10), initial_soc=10,
                )
                temporary_capacity = 10 * fraction
                store.begin_hour(fraction)
                self.assertAlmostEqual(store.soc, temporary_capacity)
                self.assertAlmostEqual(store.inaccessible_soc, 10 - temporary_capacity)
                self.assertEqual(store.charge(5), 0)

                delivered = store.discharge(min(1.0, temporary_capacity))
                self.assertAlmostEqual(store.charge(5), delivered)
                self.assertAlmostEqual(store.soc, temporary_capacity)

                # The same severity is enforced every hour, then all inaccessible
                # energy returns as soon as the fault clears.
                store.begin_hour(fraction)
                self.assertLessEqual(store.soc, temporary_capacity)
                store.begin_hour(1.0)
                self.assertAlmostEqual(store.soc, 10)
                self.assertAlmostEqual(store.inaccessible_soc, 0)


class DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weather = model.make_synthetic_weather(start="2018-01-01", years=1)
        cls.energy = model.calculate_plant_energy()

    def test_storage_sizing_closes_short_cycles(self):
        sizing = model.size_perfect_storage(self.weather, self.energy)
        self.assertTrue(sizing.feasible)
        self.assertGreater(sizing.short_capacity_kwh, 0)
        self.assertGreater(sizing.long_capacity_kwh, sizing.short_capacity_kwh)
        self.assertEqual(sizing.short_required_charge_power_kw, 0)
        self.assertEqual(sizing.long_required_charge_power_kw, 0)

    def test_initial_soc_fraction_is_explicit_for_battery_and_hydrogen(self):
        initial_fraction = 0.25
        short = model.StorageParameters(initial_soc_fraction=initial_fraction)
        for long_method in ("battery", "hydrogen"):
            long = model.StorageParameters(
                method=long_method,
                initial_soc_fraction=initial_fraction,
            )
            sizing = model.size_perfect_storage(
                self.weather, self.energy, short_storage=short, long_storage=long
            )
            self.assertAlmostEqual(
                sizing.short_initial_soc_kwh / sizing.short_capacity_kwh,
                initial_fraction,
            )
            self.assertAlmostEqual(
                sizing.long_initial_soc_kwh / sizing.long_capacity_kwh,
                initial_fraction,
            )
        with self.assertRaises(ValueError):
            model.StorageParameters(initial_soc_fraction=1.01)

    def test_low_solar_overcapacity_runs_with_integer_daily_planning(self):
        low = model.run_case(
            self.weather,
            self.weather,
            strategy=model.StrategyConfig(f_ocp=0.0, parallel_reactor_count=4),
        )
        closing = model.run_case(
            self.weather,
            self.weather,
            strategy=model.StrategyConfig(f_ocp=0.20, parallel_reactor_count=4),
        )
        self.assertIsNotNone(low.economics_perfect["lcom_usd_per_kg_ch4"])
        self.assertIsNotNone(closing.economics_perfect["lcom_usd_per_kg_ch4"])
        for result in (low, closing):
            counts = result.perfect.hourly["actual_reactor_count"]
            self.assertTrue((counts == counts.round()).all())
            self.assertTrue(counts.between(0, 4).all())

    def test_storage_transfer_power_is_unconstrained_and_reported_after_dispatch(self):
        result = model.run_case(
            self.weather,
            self.weather,
            short_storage_template=model.StorageParameters(
                max_charge_kw=0.0, max_discharge_kw=0.0,
            ),
            long_storage_template=model.StorageParameters(
                max_charge_kw=0.0, max_discharge_kw=0.0,
            ),
        )
        power = result.perfect.metadata["storage_power"]
        self.assertFalse(power["power_limits_applied"])
        self.assertGreater(power["long_required_charge_kw"], 0.0)
        self.assertEqual(
            power["long_installed_charge_kw"], power["long_required_charge_kw"],
        )

    def test_energy_capacity_factor_scales_only_long_storage(self):
        exact = model.run_case(
            self.weather,
            self.weather,
            strategy=model.StrategyConfig(f_socp_long=1.0),
        )
        three_quarters = model.run_case(
            self.weather,
            self.weather,
            strategy=model.StrategyConfig(f_socp_long=0.75),
        )
        for result in (exact, three_quarters):
            power = result.perfect.metadata["storage_power"]
            self.assertAlmostEqual(
                power["short_installed_charge_kw"], power["short_required_charge_kw"]
            )
            self.assertAlmostEqual(
                power["long_installed_charge_kw"], power["long_required_charge_kw"]
            )
        self.assertAlmostEqual(
            exact.perfect.metadata["short_storage"]["capacity_kwh"],
            exact.sizing.short_capacity_kwh,
        )
        self.assertAlmostEqual(
            three_quarters.perfect.metadata["short_storage"]["capacity_kwh"],
            three_quarters.sizing.short_capacity_kwh,
        )
        self.assertAlmostEqual(
            three_quarters.perfect.metadata["long_storage"]["capacity_kwh"],
            0.75 * three_quarters.sizing.long_capacity_kwh,
        )

    def test_dispatch_conserves_bus_energy_and_soc_bounds(self):
        result = model.run_case(self.weather, self.weather)
        hourly = result.imperfect.hourly
        self.assertLess(hourly["energy_balance_residual_kwh"].abs().max(), 1e-6)
        self.assertGreaterEqual(hourly["short_soc_kwh"].min(), -1e-9)
        self.assertLessEqual(hourly["short_soc_kwh"].max(),
                             result.imperfect.metadata["short_storage"]["capacity_kwh"] + 1e-6)

    def test_browser_dispatch_figure_keeps_hourly_resolution_and_uses_webgl(self):
        result = model.run_case(self.weather, self.weather)
        figure = model.build_dispatch_figure(result.imperfect)
        self.assertEqual(len(figure.data[0].y), len(result.imperfect.hourly))
        self.assertEqual(pd.Timestamp(figure.data[0].x0), result.imperfect.hourly.index[0])
        plotted_end = pd.Timestamp(figure.data[0].x0) + pd.Timedelta(
            milliseconds=figure.data[0].dx * (len(figure.data[0].y) - 1)
        )
        self.assertEqual(plotted_end, result.imperfect.hourly.index[-1])
        self.assertTrue(all(trace.type == "scattergl" for trace in figure.data))
        traces = {trace.name: trace for trace in figure.data}
        self.assertNotIn("Active trains", traces)
        self.assertNotIn("Planned trains", traces)
        reactor_figure = model.build_reactor_count_figure(result.imperfect)
        reactor_traces = {trace.name: trace for trace in reactor_figure.data}
        self.assertIn("Active trains", reactor_traces)
        planned_name = next(name for name in reactor_traces if name.startswith("Planned trains"))
        self.assertIn("Winter", planned_name)
        self.assertIn(reactor_traces["Active trains"].yaxis, (None, "y"))
        self.assertIn(reactor_traces[planned_name].yaxis, (None, "y"))
        self.assertEqual(reactor_figure.layout.yaxis.dtick, 1)
        self.assertIsNone(figure.layout.uirevision)

    def test_all_three_operating_strategies_run_with_constant_output(self):
        sample = self.weather.iloc[: 24 * 10]
        for short_name in ("through_night", "limping", "hard_shutdown"):
            strategy = model.StrategyConfig(short_strategy=short_name)
            sizing = model.size_perfect_storage(sample, self.energy, strategy=strategy)
            self.assertEqual(strategy.long_strategy, "constant_output")
            self.assertGreaterEqual(sizing.nominal_methane_kg_h, 0)

    def test_hard_shutdown_cools_and_reheats(self):
        strategy = model.StrategyConfig(short_strategy="hard_shutdown", parallel_reactor_count=4)
        nominal = model.nominal_methane_rate(self.weather, self.energy, model.PlantParameters(), strategy)
        summer = self.weather.loc["2018-06-01":"2018-06-03"]
        schedule = model.build_target_schedule(summer, nominal, self.energy,
                                                model.PlantParameters(), model.ThermalParameters(), strategy)
        self.assertIn("cooling", set(schedule["planned_state"]))
        self.assertGreater(schedule["startup_load_kwh"].sum(), 0)

    def test_faults_are_visible_and_reduce_output(self):
        strategy = model.StrategyConfig(parallel_reactor_count=4)
        baseline = model.run_case(self.weather, self.weather, strategy=strategy)
        fault = model.FaultEvent("hydrogen", self.weather.index[100], self.weather.index[200], 0.0)
        faulty = model.run_case(self.weather, self.weather, strategy=strategy, faults=[fault])
        window = faulty.imperfect_with_faults.hourly.iloc[100:200]
        self.assertTrue((window["hydrogen_capacity_fraction"] == 0).all())
        self.assertTrue((faulty.perfect.hourly["hydrogen_capacity_fraction"] == 1).all())
        self.assertTrue((faulty.imperfect.hourly["hydrogen_capacity_fraction"] == 1).all())
        # Storage-aware daily replanning may recover some of the deferred output
        # after the fault, so the fault-window production is the direct regression.
        self.assertLess(window["methane_kg"].sum(),
                        baseline.imperfect.hourly.iloc[100:200]["methane_kg"].sum())

    def test_zero_event_scenario_matches_fault_free_imperfect(self):
        disabled = model.FaultDistribution(mean_months_between_faults=0)
        scenario = model.FaultScenario(
            battery=disabled, hydrogen=disabled, sabatier=disabled, dac=disabled,
            seed=9,
        )
        sample = self.weather.iloc[:24 * 14]
        result = model.run_case(sample, sample, fault_scenario=scenario)
        self.assertIsNotNone(result.imperfect_with_faults)
        pd.testing.assert_frame_equal(
            result.imperfect.hourly, result.imperfect_with_faults.hourly,
        )
        self.assertEqual(
            result.imperfect.metrics["methane_total_kg"],
            result.imperfect_with_faults.metrics["methane_total_kg"],
        )

    def test_sabatier_derating_keeps_count_and_scales_throughput(self):
        sample = self.weather.iloc[:24 * 14]
        baseline = model.run_case(sample, sample)
        operating = baseline.imperfect.hourly.index[
            baseline.imperfect.hourly["methane_kg"] > 0
        ][0]
        fault = model.FaultEvent(
            "sabatier", operating, operating + pd.Timedelta(hours=1), 0.8,
        )
        result = model.run_case(sample, sample, faults=[fault])
        base_hour = result.imperfect.hourly.loc[operating]
        fault_hour = result.imperfect_with_faults.hourly.loc[operating]
        self.assertEqual(fault_hour["actual_reactor_count"], base_hour["actual_reactor_count"])
        self.assertAlmostEqual(fault_hour["methane_kg"], base_hour["methane_kg"] * 0.8)
        self.assertAlmostEqual(
            fault_hour["fresh_h2_kg"],
            fault_hour["methane_kg"] * self.energy.stoichiometry_kg_per_kg_ch4["H2"],
        )

    def test_hydrogen_and_dac_outages_cap_process_feed(self):
        sample = self.weather.iloc[:24 * 14]
        baseline = model.run_case(sample, sample)
        operating = baseline.imperfect.hourly.index[
            baseline.imperfect.hourly["methane_kg"] > 0
        ][0]
        for component, column in (
            ("hydrogen", "hydrogen_capacity_fraction"),
            ("dac", "dac_capacity_fraction"),
        ):
            with self.subTest(component=component):
                event = model.FaultEvent(
                    component, operating, operating + pd.Timedelta(hours=1), 0.0,
                )
                result = model.run_case(sample, sample, faults=[event])
                row = result.imperfect_with_faults.hourly.loc[operating]
                self.assertEqual(row[column], 0.0)
                self.assertEqual(row["methane_kg"], 0.0)

    def test_battery_fault_does_not_derate_hydrogen_inventory(self):
        sample = self.weather.iloc[:24 * 14]
        event = model.FaultEvent(
            "battery", sample.index[48], sample.index[72], 0.5,
        )
        result = model.run_case(
            sample, sample, faults=[event],
            long_storage_template=model.StorageParameters(method="hydrogen"),
        )
        window = result.imperfect_with_faults.hourly.iloc[48:72]
        self.assertTrue((window["battery_capacity_fraction"] == 0.5).all())
        short_capacity = result.imperfect_with_faults.metadata["short_storage"]["capacity_kwh"]
        self.assertTrue((window["short_soc_kwh"] <= 0.5 * short_capacity + 1e-9).all())
        self.assertTrue((window["long_inaccessible_kwh"] == 0.0).all())

    def test_fault_ratios_use_fault_free_imperfect_baseline(self):
        sample = self.weather.iloc[:24 * 14]
        event = model.FaultEvent(
            "sabatier", sample.index[24], sample.index[72], 0.8,
        )
        result = model.run_case(sample, sample, faults=[event])
        expected_production = (
            result.imperfect_with_faults.metrics["methane_total_kg"]
            / result.imperfect.metrics["methane_total_kg"]
        )
        expected_cost = (
            result.economics_imperfect_with_faults["lcom_usd_per_kg_ch4"]
            / result.economics_imperfect["lcom_usd_per_kg_ch4"]
        )
        self.assertAlmostEqual(
            result.imperfect_with_faults.metrics["relative_fault_production_ratio"],
            expected_production,
        )
        self.assertAlmostEqual(
            result.economics_imperfect_with_faults["relative_fault_cost_ratio"],
            expected_cost,
        )

    def test_parallel_reactor_rounding_is_strict_half_up(self):
        self.assertEqual(model.round_reactor_equivalents(2.3, 4), 2)
        self.assertEqual(model.round_reactor_equivalents(2.5, 4), 3)
        self.assertEqual(model.round_reactor_equivalents(2.7, 4), 3)

    def test_parallel_dispatch_never_exceeds_forecast_plan(self):
        strategy = model.StrategyConfig(parallel_reactor_count=4)
        wrong = self.weather.copy()
        wrong["capacity_factor"] = wrong["capacity_factor"].shift(12, fill_value=0.0)
        result = model.run_case(self.weather, wrong, strategy=strategy)
        hourly = result.imperfect.hourly
        self.assertTrue((hourly["actual_reactor_count"] <= hourly["planned_reactor_count"]).all())
        unit_rate = result.sizing.nominal_methane_kg_h / 4
        np.testing.assert_allclose(hourly["methane_kg"], hourly["actual_reactor_count"] * unit_rate)

    def test_better_actual_weather_respects_forecast_commitment_and_curtails(self):
        index = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
        nominal = 4.0
        strategy = model.StrategyConfig(
            short_strategy="through_night", parallel_reactor_count=4,
            reactor_scheduling_mode="daily_storage_aware",
        )
        plant = model.PlantParameters()
        thermal = model.ThermalParameters(
            calciner_ua_kw_per_k=0.0,
            carbonator_ua_kw_per_k=0.0,
            fixed_hot_auxiliary_kw=0.0,
            sabatier_reference_ua_kw_per_k=1e-12,
        )
        forecast_cf = 2.4 * self.energy.e_req_kwh_per_kg_ch4 / 10_000.0
        actual_cf = 3.1 * self.energy.e_req_kwh_per_kg_ch4 / 10_000.0
        forecast = pd.DataFrame({
            "capacity_factor": forecast_cf,
            "ambient_temperature_k": 288.15,
        }, index=index)
        actual = pd.DataFrame({
            "capacity_factor": actual_cf,
            "ambient_temperature_k": 288.15,
        }, index=index)
        result = model.simulate_dispatch(
            actual, forecast, self.energy, nominal,
            model.StorageParameters(), model.StorageParameters(),
            plant, thermal, strategy,
        )
        self.assertTrue((result.hourly["planned_reactor_count"] == 2).all())
        self.assertTrue((result.hourly["actual_reactor_count"] == 2).all())
        self.assertGreater(result.hourly["curtailed_kwh"].sum(), 0.0)
        self.assertEqual(result.metrics["forecast_reactor_ceiling_violations"], 0)

    def test_storage_can_support_ceiling_commitment(self):
        index = pd.date_range("2020-01-01", periods=24, freq="h", tz="UTC")
        nominal = 4.0
        strategy = model.StrategyConfig(
            short_strategy="through_night", parallel_reactor_count=4,
            reactor_scheduling_mode="daily_storage_aware",
        )
        plant = model.PlantParameters()
        thermal = model.ThermalParameters(
            calciner_ua_kw_per_k=0.0,
            carbonator_ua_kw_per_k=0.0,
            fixed_hot_auxiliary_kw=0.0,
            sabatier_reference_ua_kw_per_k=1e-12,
        )
        forecast_cf = 2.4 * self.energy.e_req_kwh_per_kg_ch4 / 10_000.0
        weather = pd.DataFrame({
            "capacity_factor": forecast_cf,
            "ambient_temperature_k": 288.15,
        }, index=index)
        extra_train_day = 0.6 * self.energy.e_req_kwh_per_kg_ch4 * 24
        short = model.StorageParameters(
            capacity_kwh=extra_train_day * 1.01,
            initial_soc_fraction=1.0,
        )
        result = model.simulate_dispatch(
            weather, weather, self.energy, nominal, short,
            model.StorageParameters(), plant, thermal, strategy,
        )
        self.assertTrue((result.hourly["planned_reactor_count"] == 3).all())
        self.assertTrue((result.hourly["actual_reactor_count"] == 3).all())
        self.assertEqual(result.metrics["storage_aware_round_up_days"], 1)

    def test_five_day_lookahead_preserves_storage_for_future_deficit(self):
        index = pd.date_range("2020-01-01", periods=24 * 31, freq="h", tz="UTC")
        nominal = 4.0
        plant = model.PlantParameters()
        thermal = model.ThermalParameters(
            calciner_ua_kw_per_k=0.0,
            carbonator_ua_kw_per_k=0.0,
            fixed_hot_auxiliary_kw=0.0,
            sabatier_reference_ua_kw_per_k=1e-12,
        )
        equivalents = np.full(len(index), 2.6)
        equivalents[:24] = 2.4
        weather = pd.DataFrame({
            "capacity_factor": equivalents * self.energy.e_req_kwh_per_kg_ch4 / 10_000.0,
            "ambient_temperature_k": 288.15,
        }, index=index)
        extra_train_day = 0.6 * self.energy.e_req_kwh_per_kg_ch4 * 24
        short = model.StorageParameters(
            capacity_kwh=extra_train_day * 1.01,
            initial_soc_fraction=1.0,
        )
        one_day = model.simulate_dispatch(
            weather, weather, self.energy, nominal, short,
            model.StorageParameters(), plant, thermal,
            model.StrategyConfig(
                short_strategy="through_night", parallel_reactor_count=4,
                storage_planning_lookahead_days=1,
                reactor_scheduling_mode="daily_storage_aware",
            ),
        )
        one_month = model.simulate_dispatch(
            weather, weather, self.energy, nominal, short,
            model.StorageParameters(), plant, thermal,
            model.StrategyConfig(
                short_strategy="through_night", parallel_reactor_count=4,
                storage_planning_lookahead_days=5,
                reactor_scheduling_mode="daily_storage_aware",
            ),
        )
        self.assertTrue((one_day.hourly["planned_reactor_count"].iloc[:24] == 3).all())
        self.assertTrue((one_month.hourly["planned_reactor_count"].iloc[:24] == 2).all())

    def test_storage_planning_lookahead_must_be_positive_integer(self):
        # Ten days, matching the forecast horizon the competition brief supplies.
        self.assertEqual(model.StrategyConfig().storage_planning_lookahead_days, 10)
        with self.assertRaises(ValueError):
            model.StrategyConfig(storage_planning_lookahead_days=0)
        with self.assertRaises(ValueError):
            model.StrategyConfig(storage_planning_lookahead_days=1.5)

    def test_capacity_calibration_is_capped_and_returns_best_factors(self):
        target = (0.45, 0.75)

        def fake_run_case(_actual, _forecast, *, strategy, **_kwargs):
            factors = (strategy.f_ocp, strategy.f_socp_long)
            lcom = 1.0 + sum((value - optimum) ** 2
                             for value, optimum in zip(factors, target))
            return SimpleNamespace(
                economics_perfect={"lcom_usd_per_kg_ch4": lcom},
                sizing=SimpleNamespace(
                    average_annual_balance_deficit_kwh=0.0, feasible=True,
                ),
            )

        with patch.object(model, "run_case", side_effect=fake_run_case) as mocked:
            calibrated = model.calibrate_capacity_factors(
                pd.DataFrame(), strategy=model.StrategyConfig(
                    f_ocp=0.20, f_socp_long=1.0,
                ),
                factor_minimum=0.0, factor_maximum=2.0, factor_increment=0.05,
                max_evaluations=1_000,
            )
        self.assertLessEqual(
            calibrated.evaluations, model.MAX_CAPACITY_CALIBRATION_EVALUATIONS,
        )
        self.assertEqual(mocked.call_count, calibrated.evaluations)
        self.assertEqual(calibrated.optimized_factors, target)
        self.assertTrue(calibrated.feasible)

        progress_messages = []
        with patch.object(model, "run_case", side_effect=fake_run_case):
            capped = model.calibrate_capacity_factors(
                pd.DataFrame(), factor_increment=0.05, max_evaluations=7,
                progress=progress_messages.append,
            )
        self.assertEqual(capped.evaluations, 7)
        self.assertTrue(capped.limit_reached)
        self.assertTrue(any("Best so far" in message for message in progress_messages))
        self.assertTrue(any("LCOM=$" in message for message in progress_messages))

        with patch.object(model, "run_case", side_effect=fake_run_case) as forecast_mock:
            model.calibrate_capacity_factors(
                pd.DataFrame(), pd.DataFrame(), max_evaluations=1,
            )
        candidate_kwargs = forecast_mock.call_args.kwargs
        self.assertFalse(candidate_kwargs["include_imperfect"])
        self.assertTrue(candidate_kwargs["use_forecast_commitment"])

    def test_seasonal_parallelisation_normalises_to_summer(self):
        profile = self.weather.copy()
        seasonal_cf = {
            12: 0.20, 1: 0.20, 2: 0.20,
            3: 0.40, 4: 0.40, 5: 0.40,
            6: 0.80, 7: 0.80, 8: 0.80,
            9: 0.40, 10: 0.40, 11: 0.40,
        }
        profile["capacity_factor"] = profile.index.month.map(seasonal_cf).astype(float)
        strategy = model.StrategyConfig(
            short_strategy="through_night", parallel_reactor_count=4,
            reactor_scheduling_mode="seasonal",
        )
        schedule = model.build_target_schedule(
            profile, 100.0, self.energy,
            model.PlantParameters(solar_farm_mw=100.0), model.ThermalParameters(), strategy,
        )
        monthly_counts = schedule.groupby(schedule.index.month)["planned_reactor_count"].max()
        self.assertTrue((monthly_counts.loc[[6, 7, 8]] == 4).all())
        self.assertTrue((monthly_counts.loc[[3, 4, 5, 9, 10, 11]] == 2).all())
        self.assertTrue((monthly_counts.loc[[12, 1, 2]] == 1).all())
        simulation = model.SimulationResult(
            hourly=schedule, daily=pd.DataFrame(), metrics={},
            metadata={"strategy": {"reactor_scheduling_mode": "seasonal"}},
        )
        self.assertEqual(
            model.seasonal_reactor_count_summary(simulation),
            "Winter 1 | Spring 2 | Summer 4 | Autumn 2",
        )
        figure = model.build_reactor_count_figure(simulation)
        self.assertIn("Winter 1 | Spring 2 | Summer 4 | Autumn 2", figure.data[0].name)

        one_train_schedule = model.build_target_schedule(
            profile, 100.0, self.energy,
            model.PlantParameters(solar_farm_mw=100.0), model.ThermalParameters(),
            model.StrategyConfig(
                short_strategy="through_night", parallel_reactor_count=1,
                reactor_scheduling_mode="seasonal",
            ),
        )
        one_train_monthly = one_train_schedule.groupby(
            one_train_schedule.index.month
        )["planned_reactor_count"].max()
        self.assertTrue((one_train_monthly == 1).all())

    def test_smaller_sabatier_trains_cool_faster_with_two_thirds_ua_scaling(self):
        thermal = model.ThermalParameters(sabatier_ua_scaling_exponent=2 / 3)
        c1, ua1 = model.sabatier_train_thermal_properties(100.0, thermal, 1)
        c4, ua4 = model.sabatier_train_thermal_properties(100.0, thermal, 4)
        self.assertLess(c4 / ua4, c1 / ua1)

    def test_numbered_equipment_is_split_and_sized_per_train(self):
        result = model.run_case(
            self.weather.iloc[: 24 * 10], self.weather.iloc[: 24 * 10],
            strategy=model.StrategyConfig(parallel_reactor_count=4),
        )
        rows = {row["unit_name"]: row for row in result.equipment_sizing}
        electrolyser = rows["Process electrolyser"]
        h2_per_methane = result.energy.stoichiometry_kg_per_kg_ch4["H2"]
        hourly = result.perfect.hourly
        daily_h2 = (hourly["target_methane_kg"] * h2_per_methane).resample("D").sum()
        daily_hours = (hourly["target_methane_kg"] > 0).resample("D").sum()
        expected_electrolyser_kw = (
            (daily_h2 / daily_hours.replace(0, np.nan)).fillna(0).max()
            * model.PlantParameters().electrolyser_kwh_per_kg_h2
        )
        self.assertAlmostEqual(
            electrolyser["installed_total_size"], expected_electrolyser_kw,
        )
        self.assertIn("perfect-information schedule", electrolyser["sizing_basis"])
        self.assertIn("ensures every planned day", electrolyser["sizing_basis"])
        self.assertEqual(rows["Sabatier reactor"]["count"], 4)
        self.assertAlmostEqual(
            rows["Sabatier reactor"]["installed_size_per_unit"],
            result.sizing.nominal_methane_kg_h / 4,
        )
        capex = result.economics_perfect["equipment_capex_usd"]
        self.assertIn("sabatier_reactor", capex)
        self.assertIn("feed_effluent_heat_exchanger", capex)
        self.assertNotIn("sabatier_and_heat_exchange", capex)

    def test_battery_derating_applies_to_both_stores(self):
        fault = model.FaultEvent("battery", self.weather.index[100], self.weather.index[150], 0.30)
        faulty = model.run_case(self.weather, self.weather, faults=[fault])
        window = faulty.imperfect_with_faults.hourly.iloc[100:150]
        self.assertTrue((window["battery_capacity_fraction"] == 0.30).all())
        short_capacity = faulty.imperfect_with_faults.metadata["short_storage"]["capacity_kwh"]
        long_capacity = faulty.imperfect_with_faults.metadata["long_storage"]["capacity_kwh"]
        self.assertTrue((window["short_soc_kwh"] <= 0.30 * short_capacity + 1e-9).all())
        self.assertTrue((window["long_soc_kwh"] <= 0.30 * long_capacity + 1e-9).all())
        self.assertTrue((faulty.imperfect.hourly["battery_capacity_fraction"] == 1).all())

    def test_forecast_error_changes_dispatch(self):
        wrong = self.weather.copy()
        wrong["capacity_factor"] = wrong["capacity_factor"].shift(12, fill_value=0.0)
        perfect = model.run_case(self.weather, self.weather)
        imperfect = model.run_case(self.weather, wrong)
        self.assertTrue(np.array_equal(
            imperfect.perfect.hourly["planned_reactor_count"].to_numpy(),
            imperfect.imperfect.hourly["planned_reactor_count"].to_numpy(),
        ))
        self.assertFalse(np.allclose(perfect.imperfect.hourly["methane_kg"],
                                     imperfect.imperfect.hourly["methane_kg"]))
        self.assertFalse(np.allclose(
            imperfect.imperfect.hourly["short_soc_kwh"],
            imperfect.imperfect.hourly["planned_short_soc_kwh"],
        ))

    def test_hydrogen_store_is_used_directly(self):
        sample = self.weather.iloc[:48]
        nominal = 10.0
        hydrogen = model.StorageParameters(method="hydrogen", capacity_kwh=2000,
                                            max_charge_kw=1000, max_discharge_kw=1000,
                                            initial_soc_fraction=1.0)
        empty = model.StorageParameters(capacity_kwh=0)
        sim = model.simulate_dispatch(sample, sample, self.energy, nominal, hydrogen, empty,
                                      thermal=model.ThermalParameters(calciner_ua_kw_per_k=0,
                                                                      carbonator_ua_kw_per_k=0,
                                                                      fixed_hot_auxiliary_kw=0),
                                      strategy=model.StrategyConfig(short_strategy="through_night"))
        self.assertGreater(sim.hourly["direct_h2_kg"].sum(), 0)

    def test_hydrogen_is_reserved_for_sabatier_before_fuel_cell_discharge(self):
        parameters = model.StorageParameters(
            method="hydrogen", capacity_kwh=100.0,
            max_charge_kw=100.0, max_discharge_kw=100.0,
            hydrogen_lhv_kwh_per_kg=33.33,
            hydrogen_electrolyser_kwh_per_kg=52.0,
            hydrogen_fuel_cell_kwh_per_kg=18.33,
            initial_soc_fraction=1.0,
        )
        store = model._StoreState(parameters)
        reserved_h2 = store.reservable_hydrogen_kg(1.0)
        delivered = store.discharge(
            100.0,
            reserve_carrier_kwh=reserved_h2 * parameters.hydrogen_lhv_kwh_per_kg,
        )
        used_h2, avoided_electricity = store.use_hydrogen_direct(1.0)
        self.assertAlmostEqual(used_h2, 1.0)
        self.assertAlmostEqual(avoided_electricity, 52.0)
        self.assertAlmostEqual(
            delivered,
            (100.0 - 33.33) * 18.33 / 33.33,
        )
        self.assertAlmostEqual(store.soc, 0.0, places=8)

    def test_paired_gas_store_charges_co2_first_then_allows_extra_hydrogen(self):
        ratio = 5.46
        parameters = model.StorageParameters(
            method="h2_co2", capacity_kwh=2.0 * 33.33,
            co2_capacity_kg=ratio, max_charge_kw=1000.0,
            hydrogen_lhv_kwh_per_kg=33.33,
            hydrogen_electrolyser_kwh_per_kg=52.0,
            hydrogen_fuel_cell_kwh_per_kg=18.33,
            co2_to_hydrogen_mass_ratio=ratio,
            co2_production_kwh_per_kg=2.0,
            co2_compression_kwh_per_kg=0.1,
            initial_soc_fraction=0.0,
        )
        store = model._StoreState(parameters)
        accepted = store.charge(1000.0)
        self.assertAlmostEqual(store.co2_soc_kg, ratio)
        self.assertAlmostEqual(store.soc / parameters.hydrogen_lhv_kwh_per_kg, 2.0)
        self.assertAlmostEqual(accepted, 2.0 * 52.0 + ratio * 2.1)

        stored_h2, stored_co2, avoided = store.use_feed_inventory(1.5, ratio * 1.5)
        self.assertAlmostEqual(stored_h2, 1.5)
        self.assertAlmostEqual(stored_co2, ratio)
        self.assertAlmostEqual(avoided, 1.5 * 52.0 + ratio * 2.0)
        self.assertAlmostEqual(store.soc / parameters.hydrogen_lhv_kwh_per_kg, 0.5)
        self.assertAlmostEqual(store.co2_soc_kg, 0.0)

    def test_h2_co2_case_sizes_both_vessels_from_perfect_information(self):
        sample = self.weather.iloc[:24 * 60]
        strategy = model.StrategyConfig(
            f_ocp=0.2, f_socp_long=0.5, parallel_reactor_count=3,
        )
        gas = model.StorageParameters(
            method="h2_co2", self_discharge_fraction_per_h=1e-6,
            co2_self_discharge_fraction_per_h=2e-6,
        )
        result = model.run_case(
            sample, sample, strategy=strategy, long_storage_template=gas,
        )
        required_ratio = (
            result.energy.stoichiometry_kg_per_kg_ch4["CO2"]
            / result.energy.stoichiometry_kg_per_kg_ch4["H2"]
        )
        required_h2_kg = result.sizing.long_capacity_kwh / 33.33
        self.assertAlmostEqual(
            result.sizing.long_co2_capacity_kg,
            required_h2_kg * required_ratio,
        )
        installed = result.perfect.metadata["long_storage"]
        self.assertAlmostEqual(
            installed["co2_capacity_kg"], 0.5 * result.sizing.long_co2_capacity_kg,
        )
        self.assertEqual(
            installed["co2_capacity_kg"],
            result.imperfect.metadata["long_storage"]["co2_capacity_kg"],
        )
        self.assertGreater(result.perfect.hourly["direct_co2_kg"].sum(), 0.0)
        self.assertLess(
            result.perfect.hourly["energy_balance_residual_kwh"].abs().max(), 1e-6,
        )
        self.assertAlmostEqual(
            result.economics_perfect["storage_capex_usd"],
            result.economics_imperfect["storage_capex_usd"],
        )
        unit_names = {row["unit_name"] for row in result.equipment_sizing}
        self.assertIn("Long CO2 storage vessel", unit_names)
        self.assertIn("CO2 storage charging compressor", unit_names)
        trace_names = {trace.name for trace in model.build_dispatch_figure(result.imperfect).data}
        self.assertIn("Stored H2 (kg)", trace_names)
        self.assertIn("Stored CO2 (kg)", trace_names)
        self.assertIn("Continuous methane delivery", trace_names)

    def test_hydrogen_storage_cost_uses_scaled_fuel_cell_reference_cost(self):
        hourly = pd.DataFrame(
            {"methane_kg": [1.0]},
            index=pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC"),
        )
        sim = model.SimulationResult(
            hourly, hourly, {"average_annual_methane_kg": 1.0}
        )
        hydrogen = model.StorageParameters(
            method="hydrogen", capacity_kwh=333.3,
            max_charge_kw=100.0, max_discharge_kw=500.0,
        )
        economics = model.calculate_economics(
            sim, self.energy, 0.0, model.StorageParameters(), hydrogen
        )
        expected = (
            10.0 * 600.0
            + 100.0 * 1000.0
            + 1_400_000.0 * (500.0 / 1000.0) ** 0.75
        )
        self.assertAlmostEqual(economics["storage_capex_usd"], expected)

    def test_methane_storage_smooths_output_and_is_included_in_lcom(self):
        index = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
        hourly = pd.DataFrame({"methane_kg": [0.0, 2.0, 0.0, 2.0]}, index=index)
        simulation = model.SimulationResult(
            hourly, pd.DataFrame(index=pd.date_range("2020-01-01", periods=1, tz="UTC")),
            {"average_annual_methane_kg": 8760.0},
        )
        # The vessel is no longer the default: it is what the methane storage strategy
        # is, so this has to ask for it.
        economics = model.calculate_economics(
            simulation, self.energy, 0.0,
            model.StorageParameters(), model.StorageParameters(),
            strategy=model.StrategyConfig(product_storage=True),
        )
        sizing = simulation.metadata["methane_storage"]
        self.assertAlmostEqual(sizing["continuous_delivery_kg_h"], 1.0)
        self.assertAlmostEqual(sizing["capacity_kg"], 1.0)
        self.assertAlmostEqual(sizing["initial_inventory_kg"], 1.0)
        np.testing.assert_allclose(simulation.hourly["methane_delivery_kg"], 1.0)
        np.testing.assert_allclose(
            simulation.hourly["methane_storage_inventory_kg"], [0.0, 1.0, 0.0, 1.0],
        )
        self.assertAlmostEqual(economics["methane_storage_vessel_capex_usd"], 100.0)
        # The vessel is commissioned holding 1 kg it never made, and that stock is
        # bought rather than credited to the plant as free product.
        self.assertAlmostEqual(economics["methane_initial_fill_capex_usd"], 2.0)
        self.assertAlmostEqual(
            economics["methane_storage_capex_usd"],
            economics["methane_storage_vessel_capex_usd"]
            + economics["methane_storage_compressor_capex_usd"]
            + economics["methane_storage_expander_capex_usd"]
            + economics["methane_initial_fill_capex_usd"],
        )
        self.assertAlmostEqual(
            economics["total_capex_usd"],
            economics["plant_capex_usd"] + economics["solar_capex_usd"]
            + economics["storage_capex_usd"]
            + economics["methane_storage_capex_usd"],
        )

    def test_lumpier_output_needs_a_larger_constant_delivery_buffer(self):
        # This replaces a test that starved the long store with f_SOCP and watched the
        # product buffer grow to take over the smoothing. That configuration no longer
        # exists: a plant whose seasonal store is its vessel builds no upstream store,
        # so the two can never trade duty. The physics underneath still holds, and is
        # what the vessel sizing actually rests on, so it is tested directly.
        steady = pd.Series([5.0] * 240)
        self.assertEqual(model.size_methane_storage(steady)["capacity_kg"], 0.0)

        # Same total output, delivered in ever lumpier bursts, needs an ever larger
        # tank to hold the surplus between them.
        def burst(on_hours: int) -> float:
            cycle = [10.0] * on_hours + [0.0] * on_hours
            series = pd.Series(cycle * (240 // (2 * on_hours)))
            return model.size_methane_storage(series)["capacity_kg"]

        capacities = [burst(n) for n in (2, 6, 20)]
        self.assertEqual(capacities, sorted(capacities))
        self.assertGreater(capacities[-1], capacities[0])

    def test_f_socp_builds_only_a_fraction_of_the_designed_vessel(self):
        weather = model.make_synthetic_weather("2020-01-01", years=1, latitude_deg=51.5)
        def vessel(f_socp):
            result = model.run_case(
                weather, None, include_imperfect=False,
                strategy=model.StrategyConfig(parallel_reactor_count=3,
                                              f_socp_long=f_socp, product_storage=True),
                long_storage_template=model.StorageParameters(method="h2_co2"))
            m = result.perfect.metrics
            return m["methane_storage_design_capacity_kg"], m["methane_storage_capacity_kg"]

        design_full, installed_full = vessel(1.0)
        self.assertAlmostEqual(design_full, installed_full, places=6)
        design_half, installed_half = vessel(0.5)
        # Half the tank gets built, and the shortfall has to show up as spilled
        # product rather than quietly vanishing.
        self.assertAlmostEqual(installed_half, design_half * 0.5, places=6)
        self.assertLess(installed_half, installed_full)

    def test_constant_methane_output_needs_no_buffer_capacity(self):
        output = pd.Series([3.0, 3.0, 3.0])
        self.assertEqual(model.size_methane_storage(output)["capacity_kg"], 0.0)

    def test_zero_production_economics_returns_none_lcom(self):
        hourly = pd.DataFrame({"methane_kg": [0.0]},
                              index=pd.date_range("2020-01-01", periods=1, freq="h", tz="UTC"))
        sim = model.SimulationResult(hourly, hourly, {"average_annual_methane_kg": 0.0})
        economics = model.calculate_economics(sim, self.energy, 0.0,
                                               model.StorageParameters(), model.StorageParameters())
        self.assertIsNone(economics["lcom_usd_per_kg_ch4"])

    def test_case_artifacts_are_reproducible(self):
        sample = self.weather.iloc[: 24 * 14]
        with tempfile.TemporaryDirectory() as temp:
            fault = model.FaultEvent(
                "dac", sample.index[24], sample.index[48], 0.8,
            )
            result = model.run_case(
                sample, sample, faults=[fault], save_outputs=True, output_root=temp,
            )
            output = Path(temp) / result.case_id
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["case_id"], result.case_id)
            self.assertTrue((output / "imperfect_hourly.csv").exists())
            self.assertTrue((output / "imperfect_with_faults_hourly.csv").exists())
            self.assertEqual(summary["fault_events"][0]["component"], "dac")
            self.assertTrue((output / "dispatch.html").exists())

    def test_sizing_profile_builds_a_separate_plant_for_the_forecast_cases(self):
        # Two distinct periods, so a plant sized on the first is genuinely not the
        # plant a perfect-information study of the second would have specified.
        weather = model.make_synthetic_weather("2014-01-01", years=4, latitude_deg=51.5)
        training, actual = model.split_weather_period(
            weather, training_years=2, evaluation_years=2)
        forecast = model.build_climatology_forecast(training, actual)
        strategy = model.StrategyConfig(parallel_reactor_count=4)

        shared = model.run_case(actual, forecast, include_baseline=True,
                                strategy=strategy)
        split = model.run_case(actual, forecast, include_baseline=True,
                               strategy=strategy, sizing_profile=training)

        # Omitting sizing_profile must leave the model exactly as it was.
        self.assertIsNone(shared.sizing_imperfect)
        self.assertEqual(shared.imperfect_realised_deficit_kwh_per_year, 0.0)

        # With it, the forecast-driven cases run their own plant while perfect
        # information keeps the one sized on the decade it actually meets.
        self.assertIsNotNone(split.sizing_imperfect)
        self.assertEqual(split.sizing.long_capacity_kwh, shared.sizing.long_capacity_kwh)
        self.assertNotEqual(split.sizing_imperfect.nominal_methane_kg_h,
                            split.sizing.nominal_methane_kg_h)
        # Perfect information is untouched by the split; only the other cases move.
        self.assertAlmostEqual(split.economics_perfect["lcom_usd_per_kg_ch4"],
                               shared.economics_perfect["lcom_usd_per_kg_ch4"], places=9)
        self.assertNotAlmostEqual(split.economics_imperfect["lcom_usd_per_kg_ch4"],
                                  shared.economics_imperfect["lcom_usd_per_kg_ch4"],
                                  places=6)
        # Capital differs because the two plants are different plants.
        self.assertNotAlmostEqual(split.economics_imperfect["total_capex_usd"],
                                  split.economics_perfect["total_capex_usd"], places=2)
        # The baseline is a bolt-on to the built plant, so it follows its throughput.
        self.assertIsNotNone(split.economics_baseline)

    def test_realised_deficit_reports_a_store_that_was_under_built(self):
        weather = model.make_synthetic_weather("2014-01-01", years=4, latitude_deg=51.5)
        training, actual = model.split_weather_period(
            weather, training_years=2, evaluation_years=2)
        energy = model.calculate_plant_energy()
        plant, thermal = model.PlantParameters(), model.ThermalParameters()
        strategy = model.StrategyConfig(parallel_reactor_count=4)
        sizing = model.size_perfect_storage(
            training, energy, plant, thermal, model.StorageParameters(),
            model.StorageParameters(self_discharge_fraction_per_h=1e-5), strategy)
        short, _ = model._configured_store(
            model.StorageParameters(), sizing.short_capacity_kwh,
            sizing.short_initial_soc_kwh, 1.0)
        long_store, _ = model._configured_store(
            model.StorageParameters(self_discharge_fraction_per_h=1e-5),
            sizing.long_capacity_kwh, sizing.long_initial_soc_kwh, strategy.f_socp_long)

        # Measured against the record it was sized from, a successful sizing closes
        # its cycle by construction.
        on_its_own_record = model.realised_cyclic_deficit(
            training, sizing.nominal_methane_kg_h, energy, plant, thermal, strategy,
            short, long_store)
        self.assertLessEqual(on_its_own_record, 1e-6)

        # Starve the store and the same weather must report a deficit, which is what
        # makes this able to catch an under-built plant at all.
        starved, _ = model._configured_store(
            model.StorageParameters(self_discharge_fraction_per_h=1e-5),
            sizing.long_capacity_kwh * 0.01, 0.0, strategy.f_socp_long)
        self.assertGreater(
            model.realised_cyclic_deficit(training, sizing.nominal_methane_kg_h * 4,
                                          energy, plant, thermal, strategy, short,
                                          starved),
            0.0)


if __name__ == "__main__":
    unittest.main()
