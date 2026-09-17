import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import app as dashboard


class DashSmokeTests(unittest.TestCase):
    def setUp(self):
        dashboard.SHOWCASE_RESULTS.clear()

    def test_app_serves_layout_and_registers_callbacks(self):
        application = dashboard.create_app()
        response = application.server.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(application.callback_map), 4)
        layout = application.server.test_client().get("/_dash-layout")
        layout_data = layout.get_json()
        layout_json = json.dumps(layout_data)
        self.assertNotIn('"f-pocp"', layout_json)
        self.assertIn('"factor-range-min"', layout_json)
        self.assertIn('"factor-range-max"', layout_json)
        self.assertIn('"factor-range-step"', layout_json)
        self.assertIn('"capacity-setting-mode"', layout_json)
        self.assertIn('"calibrated-factors"', layout_json)
        # Auto-calibration is withdrawn from the dashboard but kept in the codebase.
        self.assertNotIn("Auto-calibrate capacities", layout_json)
        self.assertTrue(hasattr(dashboard, "calibrate_capacity_factors"))
        self.assertIn('"information-mode"', layout_json)
        self.assertIn('"ninja-api-key"', layout_json)
        self.assertIn('"ninja-api-key-warning"', layout_json)
        self.assertIn("Renewables.ninja requires an API key", layout_json)
        self.assertIn('"result-category"', layout_json)
        self.assertIn('"fault-enabled"', layout_json)
        self.assertIn('"fault-seed"', layout_json)
        for component in ("battery", "hydrogen", "sabatier", "dac"):
            self.assertIn(f'"fault-{component}-months"', layout_json)
            self.assertIn(f'"fault-{component}-duration"', layout_json)
            self.assertIn(f'"fault-{component}-capacity"', layout_json)
        self.assertIn("one event every N months", layout_json)
        self.assertIn('"reactor-count"', layout_json)
        self.assertIn('"reactor-scheduling-mode"', layout_json)
        self.assertIn("Seasonal parallelisation", layout_json)
        self.assertIn('"slow-calibration-warning"', layout_json)
        self.assertIn("reactor parallelisation is an experimental feature", layout_json)
        self.assertIn('"run-specific"', layout_json)
        self.assertIn('"run-all"', layout_json)
        self.assertIn('"load-dispatch"', layout_json)
        self.assertIn('"reactor-count-graph"', layout_json)
        self.assertIn('"initial-soc"', layout_json)
        self.assertIn('"comparison-metric"', layout_json)
        self.assertIn('"comparison-bar"', layout_json)
        self.assertIn('"comparison-storage-filter"', layout_json)
        self.assertIn('"comparison-strategy-filter"', layout_json)
        self.assertIn('"comparison-case-filter"', layout_json)
        self.assertIn('"saved-assumptions"', layout_json)
        self.assertIn('"assumptions-table"', layout_json)
        self.assertIn("System & economic parameters", layout_json)
        self.assertIn("Sizing", layout_json)
        self.assertIn('"sizing-content"', layout_json)
        self.assertIn("Save parameters", layout_json)
        self.assertIn('"type": "Slider"', layout_json)

        self.assertIn("can take several minutes", layout_json)
        self.assertIn('"persistence_type": "local"', layout_json)

        def component_with_id(node, component_id):
            if isinstance(node, dict):
                props = node.get("props", {})
                if props.get("id") == component_id:
                    return node
                for value in node.values():
                    found = component_with_id(value, component_id)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for value in node:
                    found = component_with_id(value, component_id)
                    if found is not None:
                        return found
            return None

        ninja_key_input = component_with_id(layout_data, "ninja-api-key")
        self.assertEqual(ninja_key_input["props"]["type"], "password")
        self.assertNotIn("persistence", ninja_key_input["props"])

        # The capacity-mode control stays so every callback still resolves, offering
        # only manual. Persistence must be off too, or a browser that saved "auto"
        # before the withdrawal would go on calibrating with nothing on screen to say so.
        mode = component_with_id(layout_data, "capacity-setting-mode")
        self.assertEqual([option["value"] for option in mode["props"]["options"]],
                         ["manual"])
        self.assertEqual(mode["props"]["value"], "manual")
        self.assertNotIn("persistence", mode["props"])

        reactor_slider = component_with_id(layout_data, "reactor-count")
        self.assertIsNotNone(reactor_slider)
        self.assertEqual(reactor_slider["type"], "Slider")
        self.assertEqual(
            (reactor_slider["props"]["min"], reactor_slider["props"]["max"],
             reactor_slider["props"]["step"]),
            (1, 6, 1),
        )

    def test_daily_auto_calibration_warns_but_seasonal_does_not(self):
        warning = dashboard._slow_calibration_warning("daily_storage_aware", "auto")
        self.assertIn("very slow", warning)
        self.assertEqual(dashboard._slow_calibration_warning("seasonal", "auto"), "")
        self.assertEqual(
            dashboard._slow_calibration_warning("daily_storage_aware", "manual"), "",
        )

    def test_factor_slider_scale_has_generous_defaults_and_validates_edits(self):
        minimum, maximum, increment, marks = dashboard._slider_scale(0, 2, 0.05)
        self.assertEqual((minimum, maximum, increment), (0, 2, 0.05))
        self.assertEqual(marks, {0: "0", 0.5: "0.5", 1.0: "1", 1.5: "1.5", 2.0: "2"})
        with self.assertRaises(ValueError):
            dashboard._slider_scale(2, 1, 0.05)
        with self.assertRaises(ValueError):
            dashboard._slider_scale(0, 2, 0)

    def test_auto_calibration_uses_winner_for_final_case_and_job(self):
        job_id = uuid4().hex
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS[job_id] = {
                "status": "queued", "messages": ["Run queued."],
                "delivered_selection": None, "delivered_failure": False,
            }
        optimized_strategy = dashboard.StrategyConfig(
            short_strategy="limping", f_ocp=0.45,
            f_socp_long=0.75,
            parallel_reactor_count=4,
        )
        calibration = SimpleNamespace(
            strategy=optimized_strategy,
            optimized_factors=(0.45, 0.75),
            starting_factors=(0.20, 1.0),
            starting_lcom_usd_per_kg_ch4=2.0,
            optimized_lcom_usd_per_kg_ch4=1.5,
            evaluations=23, converged=True, limit_reached=False, feasible=True,
            average_annual_balance_deficit_kwh=0.0,
        )
        final_strategies = []

        def fake_run_case(*_args, strategy, include_imperfect, **_kwargs):
            final_strategies.append(strategy)
            perfect = SimpleNamespace(metadata={}, warnings=[])
            imperfect = SimpleNamespace(metadata={}, warnings=[]) if include_imperfect else None
            return SimpleNamespace(perfect=perfect, imperfect=imperfect)

        parameters = {
            "lat": 51.5074, "lon": -0.1278, "farm_mw": 10,
            "weather_source": "demo", "latest_year": 2025,
            "reactor_count": 4, "reactor_scheduling_mode": "daily_storage_aware",
            "information_mode": "comparison", "run_scope": "specific",
            "short_name": "limping", "long_method": "battery",
            "capacity_setting_mode": "auto",
            "f_ocp": 0.2, "f_socp": 1.0,
            "factor_minimum": 0.0, "factor_maximum": 2.0, "factor_increment": 0.05,
            "initial_soc_fraction": 0.5, "fault_name": "none",
        }
        with patch.object(
                dashboard, "calibrate_capacity_factors", return_value=calibration
        ) as calibrator, patch.object(
                dashboard, "run_case", side_effect=fake_run_case
        ), patch.object(
                dashboard, "_compact_dashboard_result", return_value=(1.0, 1.0)
        ):
            dashboard._execute_single_site_job(job_id, parameters)
        job = dashboard._job_snapshot(job_id)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["optimized_factors"]["f_ocp"], 0.45)
        self.assertEqual(final_strategies, [optimized_strategy])
        self.assertEqual(
            calibrator.call_args.kwargs["max_evaluations"],
            dashboard.MAX_CAPACITY_CALIBRATION_EVALUATIONS,
        )
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS.pop(job_id, None)

    def test_stochastic_fault_job_keeps_three_categories_and_fault_bands(self):
        job_id = uuid4().hex
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS[job_id] = {
                "status": "queued", "messages": ["Run queued."],
                "delivered_selection": None, "delivered_failure": False,
            }
        disabled = {"months": 0, "duration_h": 48, "capacity_fraction": 0.8}
        dashboard._execute_single_site_job(job_id, {
            "lat": 51.5074, "lon": -0.1278, "farm_mw": 10,
            "weather_source": "demo", "latest_year": 2025,
            "reactor_count": 2, "reactor_scheduling_mode": "seasonal",
            "information_mode": "comparison", "run_scope": "specific",
            "short_name": "limping", "long_method": "battery",
            "capacity_setting_mode": "manual", "f_ocp": 0.2, "f_socp": 1.0,
            "initial_soc_fraction": 0.5, "faults_enabled": True, "fault_seed": 0,
            "fault_config": {
                "battery": disabled, "hydrogen": disabled, "dac": disabled,
                "sabatier": {"months": 1, "duration_h": 48,
                              "capacity_fraction": 0.8},
            },
        })
        job = dashboard._job_snapshot(job_id)
        self.assertEqual(job["status"], "complete")
        result = next(iter(job["results"].values()))
        self.assertIsNotNone(result.imperfect_with_faults)
        for simulation in (result.perfect, result.imperfect, result.imperfect_with_faults):
            summary = simulation.metadata["pfd_summary"]
            self.assertEqual(len(summary["streams"]), 12)
            self.assertAlmostEqual(summary["streams"]["S06"]["flow_kg_h"],
                                   simulation.hourly["methane_delivery_kg"].mean())
        self.assertGreater(result.imperfect_with_faults.metrics["sabatier_event_count"], 0)
        self.assertEqual(
            list(result.imperfect_with_faults.hourly.columns),
            list(dashboard.DASHBOARD_HOURLY_COLUMNS),
        )
        figure = dashboard.build_dispatch_figure(result.imperfect_with_faults)
        self.assertGreater(len(figure.layout.shapes), 0)
        labels = {
            card.children[0].children
            for card in dashboard._result_cards(
                result, job["evaluation_period"], "comparison",
                result_category="imperfect_with_faults",
            )
        }
        self.assertIn("Fault cost ratio", labels)
        self.assertIn("Incremental fault loss", labels)
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS.pop(job_id, None)

    def test_assumption_rows_cover_system_and_cost_inputs(self):
        rows = dashboard.build_assumption_rows()
        keys = {row["key"] for row in rows if row.get("editable")}
        self.assertIn("plant.electrolyser_kwh_per_kg_h2", keys)
        self.assertIn("plant.hydrogen_lhv_kwh_per_kg", keys)
        self.assertIn("plant.methane_molar_mass_kg_per_mol", keys)
        self.assertIn("thermal.calciner_ua_kw_per_k", keys)
        self.assertIn("thermal.sabatier_ua_scaling_exponent", keys)
        self.assertIn("strategy.storage_planning_lookahead_days", keys)
        self.assertIn("plant.fuel_cell_kwh_per_kg_h2", keys)
        self.assertIn("storage.long_hydrogen.self_discharge_fraction_per_h", keys)
        self.assertIn("storage.long_h2_co2.co2_self_discharge_fraction_per_h", keys)
        self.assertIn("economic.co2_storage.tank_capex_usd_per_kg", keys)
        self.assertIn("weather.system_loss", keys)
        self.assertIn("synthetic.maximum_capacity_factor", keys)
        self.assertNotIn("fault.start_offset_days", keys)
        self.assertNotIn("fault.electrolyser_availability", keys)
        self.assertIn("economic.financial.real_discount_rate", keys)
        self.assertIn(
            "economic.plant_units.sabatier_reactor.scaling_exponent", keys
        )
        self.assertIn(
            "economic.plant_units.feed_effluent_heat_exchanger.scaling_exponent", keys
        )
        self.assertTrue(all(row["unit"] and row["description"] for row in rows))
        plant_fields = set(dashboard.PlantParameters.__dataclass_fields__)
        documented_plant_fields = (
            set(dashboard.PLANT_ASSUMPTION_META)
            | set(dashboard.PHYSICAL_PROPERTY_ASSUMPTION_META)
        )
        self.assertEqual(documented_plant_fields, plant_fields)
        self.assertEqual(
            set(dashboard.SYNTHETIC_WEATHER_ASSUMPTION_META),
            set(dashboard.SyntheticWeatherParameters.__dataclass_fields__),
        )
        self.assertEqual(
            set(dashboard.THERMAL_ASSUMPTION_META),
            set(dashboard.ThermalParameters.__dataclass_fields__),
        )
        self.assertEqual(
            set(dashboard.STRATEGY_ASSUMPTION_META),
            set(dashboard.StrategyConfig.__dataclass_fields__),
        )
        self.assertEqual(
            set(dashboard.WEATHER_ASSUMPTION_META),
            set(dashboard.WeatherConfig.__dataclass_fields__) - {"lat", "lon", "latest_year"},
        )
        self.assertEqual(
            set(dashboard.STORAGE_ASSUMPTION_META),
            set(dashboard.StorageParameters.__dataclass_fields__)
            - {
                "method", "capacity_kwh", "max_charge_kw", "max_discharge_kw",
                "hydrogen_lhv_kwh_per_kg", "hydrogen_electrolyser_kwh_per_kg",
                "hydrogen_fuel_cell_kwh_per_kg",
                "co2_capacity_kg", "co2_to_hydrogen_mass_ratio",
                "co2_production_kwh_per_kg", "co2_compression_kwh_per_kg",
                "hydrogen_compression_kwh_per_kg", "hydrogen_expansion_kwh_per_kg",
                "co2_expansion_kwh_per_kg",
                "hydrogen_expansion_reheat_kwh_per_kg",
                "co2_expansion_reheat_kwh_per_kg",
                "hydrogen_charge_capacity_kw",
                "hydrogen_fuel_cell_capacity_kw",
                "hydrogen_storage_compressor_capacity_kw",
                "hydrogen_storage_expander_capacity_kw",
                "co2_charge_compressor_capacity_kw",
                "co2_storage_expander_capacity_kw",
            },
        )

        costs = json.loads(dashboard.Path("costs.json").read_text(encoding="utf-8"))
        expected_economic_keys = {"metadata.currency", "metadata.base_year"}

        def collect(prefix, value):
            if isinstance(value, dict):
                for name, child in value.items():
                    collect(f"{prefix}.{name}" if prefix else name, child)
            elif not prefix.startswith("metadata."):
                expected_economic_keys.add(prefix)

        collect("", costs)
        documented_economic_keys = {
            key.removeprefix("economic.")
            for key in keys if key.startswith("economic.")
        }
        self.assertEqual(documented_economic_keys, expected_economic_keys)

    def test_synthetic_weather_parameters_change_offline_profile(self):
        parameters = dashboard.SyntheticWeatherParameters(
            maximum_capacity_factor=0.10,
            ambient_mean_temperature_k=300.0,
            ambient_seasonal_amplitude_k=0.0,
            ambient_diurnal_amplitude_k=0.0,
        )
        weather = dashboard.make_synthetic_weather(
            "2015-01-01", years=1, parameters=parameters
        )
        self.assertLessEqual(weather["capacity_factor"].max(), 0.10)
        self.assertTrue((weather["ambient_temperature_k"] == 300.0).all())

    def test_ninja_api_key_prefers_env_then_falls_back_to_entry(self):
        with patch.object(dashboard, "_load_api_token", return_value="env-key"):
            self.assertEqual(dashboard._resolve_ninja_api_key("entered-key"), "env-key")
        with patch.object(dashboard, "_load_api_token", return_value=None):
            self.assertEqual(dashboard._resolve_ninja_api_key(" entered-key "), "entered-key")
            self.assertIsNone(dashboard._resolve_ninja_api_key("  "))

    def test_ninja_run_without_any_api_key_reports_a_clear_warning(self):
        job_id = uuid4().hex
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS[job_id] = {
                "status": "queued", "messages": ["Run queued."],
            }
        with patch.object(dashboard, "_load_api_token", return_value=None):
            dashboard._execute_single_site_job(job_id, {
                "lat": 51.5074, "lon": -0.1278,
                "weather_source": "ninja", "information_mode": "comparison",
            })
        job = dashboard._job_snapshot(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("requires an API key", job["error"])
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS.pop(job_id, None)

    def test_single_site_callback_reports_required_storage_power(self):
        application = dashboard.create_app()
        job_id = uuid4().hex
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS[job_id] = {
                "status": "queued", "messages": ["Run queued."],
                "delivered_selection": None, "delivered_failure": False,
            }
        dashboard._execute_single_site_job(job_id, {
            "lat": 51.5074, "lon": -0.1278, "farm_mw": 10,
            "weather_source": "demo", "latest_year": 2025,
            "reactor_count": 4,
            "information_mode": "comparison",
            "run_scope": "all", "short_name": "limping", "long_method": "battery",
            "f_ocp": 0.2, "f_socp": 1.0,
            "initial_soc_fraction": 0.35,
            "fault_name": "none",
        })
        job = dashboard._job_snapshot(job_id)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(len(job["results"]),
                         len(dashboard.OPERATING_STRATEGIES) * len(dashboard.LONG_STORAGE_METHODS))
        self.assertEqual(
            set(job["results"]),
            {dashboard._result_key(short, storage)
             for short in dashboard.OPERATING_STRATEGIES
             for storage in dashboard.LONG_STORAGE_METHODS},
        )
        self.assertTrue(any("unconstrained" in message for message in job["messages"]))
        self.assertTrue(any("dispatch progress" in message.casefold()
                            for message in job["messages"]))
        self.assertTrue(any("Economics" in message for message in job["messages"]))
        total = len(dashboard.OPERATING_STRATEGIES) * len(dashboard.LONG_STORAGE_METHODS)
        self.assertTrue(any(f"Case {total}/{total}" in message
                            for message in job["messages"]))
        for result in job["results"].values():
            self.assertEqual(
                list(result.perfect.hourly.columns),
                list(dashboard.DASHBOARD_HOURLY_COLUMNS),
            )
            self.assertEqual(
                list(result.imperfect.hourly.columns),
                list(dashboard.DASHBOARD_HOURLY_COLUMNS),
            )
            self.assertAlmostEqual(
                result.imperfect.metadata["short_storage"]["initial_soc_fraction"], 0.35
            )
            self.assertAlmostEqual(
                result.imperfect.metadata["long_storage"]["initial_soc_fraction"], 0.35
            )
            self.assertTrue(result.equipment_sizing)
            for simulation in (result.perfect, result.imperfect):
                summary = simulation.metadata["pfd_summary"]
                self.assertEqual(summary["hours"], len(simulation.hourly))
                self.assertEqual(summary["storage_method"], simulation.metadata["long_storage"]["method"])
                self.assertTrue(all(stream["flow_kg_h"] is not None for stream in summary["streams"].values()))
                self.assertAlmostEqual(summary["streams"]["S06"]["flow_kg_h"],
                                       simulation.hourly["methane_delivery_kg"].mean())
            reactor = next(row for row in result.equipment_sizing
                           if row["unit_name"] == "Sabatier reactor")
            self.assertEqual(reactor["count"], 4)
        comparison = dashboard.build_case_comparison_figure(
            job, "lcom_usd_per_kg_ch4"
        )
        # Bars, the perfect-information diamonds, and the no-storage baseline that
        # single-site runs now compute as a reference point.
        self.assertEqual(len(comparison.data), 3)
        baseline = next(trace for trace in comparison.data
                        if "Baseline" in (trace.name or ""))
        self.assertTrue(all(value is not None for value in baseline.y))
        self.assertEqual(len(comparison.data[0].x), total)
        self.assertEqual(len(comparison.data[0].y), total)
        self.assertEqual(len(comparison.data[1].y), total)
        self.assertEqual(comparison.data[1].name, "Perfect information, no faults")
        self.assertEqual(len(set(comparison.data[0].marker.color)), total)
        battery_only = dashboard.build_case_comparison_figure(
            job, "lcom_usd_per_kg_ch4", selected_storage=["battery"]
        )
        self.assertEqual(len(battery_only.data[0].x), 3)
        self.assertTrue(all("Battery" in label for label in battery_only.data[0].x))
        reactor_figure = dashboard.build_active_reactor_figure(
            job["results"][dashboard._result_key("limping", "battery")].imperfect
        )
        self.assertEqual(len(reactor_figure.data), 2)
        self.assertTrue(reactor_figure.data[0].name.startswith("Forecast-planned trains"))
        self.assertIn("Winter", reactor_figure.data[0].name)
        self.assertEqual(reactor_figure.data[1].name, "Actual active trains")
        self.assertTrue(all(float(value).is_integer()
                            for value in reactor_figure.data[1].y))

        key = next(name for name in application.callback_map if name.startswith("..run-status"))
        callback = application.callback_map[key]["callback"]
        runner = getattr(callback, "__wrapped__", callback)
        status, cards, selected_disabled, all_disabled, plot_disabled, calibrated = runner(
            1, "limping", "battery", "manual", "imperfect", {"job_id": job_id}
        )
        labels = {card.children[0].children for card in cards}
        self.assertFalse(selected_disabled)
        self.assertFalse(all_disabled)
        self.assertFalse(plot_disabled)
        self.assertIs(calibrated, dashboard.no_update)
        self.assertEqual(job["evaluation_period"], "2015")
        self.assertIn("Evaluation period", labels)
        self.assertIn("Operating strategy", labels)
        self.assertIn("Long storage", labels)
        self.assertIn("Short energy: installed / required", labels)
        self.assertIn("Long energy: installed / required", labels)
        self.assertIn("Short power: installed / required", labels)
        self.assertIn("Long power: installed / required", labels)
        self.assertIn("Annual cyclic energy deficit", labels)
        # This case runs battery long storage, which has no gate buffer any more, so
        # the gate flow is whatever was made that hour rather than a committed rate.
        # Only the methane storage strategy reports a continuous delivery.
        self.assertIn("Mean methane delivery (floating)", labels)
        self.assertNotIn("Continuous methane delivery", labels)
        self.assertIn("Product methane buffer vessel", labels)
        self.assertIn("Methane buffer CAPEX", labels)
        self.assertNotIn("open", status.children[1].to_plotly_json()["props"])
        _, switched_cards, _, _, switched_plot_disabled, _ = runner(
            2, "hard_shutdown", "hydrogen", "manual", "imperfect", {"job_id": job_id}
        )
        switched_values = {card.children[0].children: card.children[1].children
                           for card in switched_cards}
        self.assertEqual(switched_values["Operating strategy"], "Hard Shutdown")
        self.assertEqual(switched_values["Long storage"], "Hydrogen")
        self.assertFalse(switched_plot_disabled)
        _, _, _, auto_all_disabled, _, _ = runner(
            3, "hard_shutdown", "hydrogen", "auto", "imperfect", {"job_id": job_id}
        )
        self.assertTrue(auto_all_disabled)
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS.pop(job_id, None)

    def test_perfect_only_mode_skips_forecast_dispatch(self):
        application = dashboard.create_app()
        assumptions = dashboard.default_assumption_config()
        assumptions["plant.electrolyser_kwh_per_kg_h2"] = 60.0
        assumptions["plant.hydrogen_molar_mass_kg_per_mol"] = 0.004
        assumptions["storage.short_battery.self_discharge_fraction_per_h"] = 0.001
        assumptions["economic.solar.capex_usd_per_kw"] = 1234.0
        job_id = uuid4().hex
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS[job_id] = {
                "status": "queued", "messages": ["Run queued."],
                "delivered_selection": None, "delivered_failure": False,
            }
        dashboard._execute_single_site_job(job_id, {
            "lat": 51.5074, "lon": -0.1278, "farm_mw": 10,
            "weather_source": "demo", "latest_year": 2025,
            "reactor_count": 4,
            "information_mode": "perfect_only",
            "run_scope": "specific", "short_name": "limping", "long_method": "battery",
            "f_ocp": 0.2, "f_socp": 1.0,
            "fault_name": "none", "assumptions": assumptions,
        })
        job = dashboard._job_snapshot(job_id)
        self.assertEqual(job["status"], "complete")
        self.assertEqual(len(job["results"]), 1)
        self.assertTrue(all(result.imperfect is None for result in job["results"].values()))
        self.assertTrue(all(result.economics_imperfect is None
                            for result in job["results"].values()))
        self.assertTrue(any("forecast case skipped" in message for message in job["messages"]))
        selected = job["results"][dashboard._result_key("limping", "battery")]
        self.assertAlmostEqual(
            selected.perfect.metadata["short_storage"]["self_discharge_fraction_per_h"],
            0.001,
        )
        self.assertAlmostEqual(selected.economics_perfect["solar_capex_usd"], 12_340_000)
        expected_hydrogen = (
            4 * assumptions["plant.hydrogen_molar_mass_kg_per_mol"]
            / assumptions["plant.methane_molar_mass_kg_per_mol"]
        )
        self.assertAlmostEqual(
            selected.energy.stoichiometry_kg_per_kg_ch4["H2"], expected_hydrogen
        )
        self.assertEqual(
            job["assumptions"]["plant.electrolyser_kwh_per_kg_h2"], 60.0
        )
        self.assertEqual(
            list(selected.perfect.hourly.columns),
            list(dashboard.DASHBOARD_HOURLY_COLUMNS),
        )
        labels = {
            card.children[0].children
            for card in dashboard._result_cards(
                selected, job["evaluation_period"], "perfect_only"
            )
        }
        self.assertIn("Annual cyclic energy deficit", labels)
        self.assertNotIn("Forecast cost ratio", labels)
        key = next(name for name in application.callback_map if name.startswith("..run-status"))
        callback = application.callback_map[key]["callback"]
        runner = getattr(callback, "__wrapped__", callback)
        _, selected_cards, _, _, _, _ = runner(
            1, "limping", "battery", "manual", "perfect", {"job_id": job_id}
        )
        missing_status, missing_cards, _, _, _, _ = runner(
            2, "hard_shutdown", "hydrogen", "manual", "perfect", {"job_id": job_id}
        )
        self.assertTrue(selected_cards)
        self.assertFalse(missing_cards)
        self.assertIn("was not calculated", missing_status.children)
        _, restored_cards, _, _, _, _ = runner(
            3, "limping", "battery", "manual", "perfect", {"job_id": job_id}
        )
        self.assertTrue(restored_cards)
        with dashboard.RUN_JOBS_LOCK:
            dashboard.RUN_JOBS.pop(job_id, None)

    def test_showcase_map_starts_without_result_dots(self):
        figure = dashboard.build_showcase_map("plant_utilisation")
        plotted = sum(len(trace.lat) for trace in figure.data)
        self.assertEqual(plotted, 0)
        self.assertEqual(len(dashboard.build_showcase_map(clickable=True).data), 1)

    def test_showcase_selection_toggles_and_results_replace_existing_dot(self):
        original = dashboard.load_showcase().iloc[0].to_dict()
        click = {"points": [{"lat": original["lat"], "lon": original["lon"],
                              "hovertext": original["site"]}]}
        selected = dashboard.toggle_showcase_location(click, [])
        self.assertEqual(selected[0]["site"], original["site"])
        self.assertEqual(dashboard.toggle_showcase_location(click, selected), [])
        custom = dashboard.toggle_showcase_location({"points": [{"lat": 48.5, "lon": 2.5}]}, [])
        self.assertEqual(custom[0]["site"], "48.50, 2.50")
        replacement = {**original, "lcom_usd_per_kg_ch4": 123.0}
        records = [replacement, {**original, **custom[0]}]
        data = dashboard.showcase_data(records)
        self.assertEqual(len(data), 2)
        self.assertEqual(data.loc[data.site == original["site"], "lcom_usd_per_kg_ch4"].iloc[0], 123)
        figure = dashboard.build_showcase_map(records=records, selected=selected, clickable=True)
        self.assertGreater(len(figure.data[0].lat), 7000)
        self.assertEqual(len(figure.data[-1].lat), 2)
        self.assertEqual(list(figure.data[1].hovertext), [original["site"]])

    def test_showcase_batch_runs_locations_and_survives_a_failed_site(self):
        for information_mode in ("perfect_only", "comparison"):
            with self.subTest(information_mode=information_mode):
                job_id = uuid4().hex
                dashboard.RUN_JOBS[job_id] = {"status": "queued", "messages": ["Queued"]}
                locations = [{"site": "Paris", "lat": 48.5, "lon": 2.5},
                             {"site": "Failed site", "lat": 49.0, "lon": 3.0},
                             {"site": "Madrid", "lat": 40.5, "lon": -3.5}]
                parameters = {
                    "farm_mw": 10, "weather_source": "demo", "latest_year": 2025,
                    "reactor_count": 2, "reactor_scheduling_mode": "seasonal",
                    "information_mode": information_mode, "run_scope": "investigate",
                    "short_name": "limping", "long_method": "battery",
                    "f_ocp": 0.2, "f_socp": 0.75, "initial_soc_fraction": 0.5,
                    "faults_enabled": information_mode == "comparison", "fault_seed": 42,
                }
                actual_worker = dashboard._execute_single_site_job
                calls = []

                def worker(child_id, values, progress_callback=None):
                    calls.append(values)
                    if values["site"] == "Failed site":
                        raise ValueError("Weather unavailable")
                    actual_worker(child_id, values, progress_callback)

                try:
                    with patch.object(dashboard, "_execute_single_site_job", side_effect=worker):
                        dashboard._execute_showcase_job(job_id, locations, parameters)
                    job = dashboard._job_snapshot(job_id)
                    self.assertEqual(job["status"], "complete")
                    self.assertEqual([row["site"] for row in job["rows"]], ["Paris", "Madrid"])
                    self.assertEqual(job["errors"], ["Failed site: Weather unavailable"])
                    self.assertEqual(list(dashboard.SHOWCASE_RESULTS.values()), job["rows"])
                    self.assertTrue(all(call["run_scope"] == "specific" for call in calls))
                    self.assertTrue(all(call["reactor_count"] == 2 for call in calls))
                    self.assertFalse(any(value.get("internal") for value in dashboard.RUN_JOBS.values()))
                    json.dumps(job["rows"], allow_nan=False)
                    # Showcase runs force comparison mode and faults on regardless of
                    # the requested information mode, so every row carries all three
                    # variants and the map never has an unpopulated metric.
                    self.assertTrue(all(call["information_mode"] == "comparison"
                                        for call in calls))
                    self.assertTrue(all(call["faults_enabled"] for call in calls))
                    for row in job["rows"]:
                        self.assertGreater(row["average_annual_methane_kg"], 0)
                        self.assertIn("comparison", row["data_status"])
                        economics = row["outputs"]
                        expected = (economics["economics_imperfect_with_faults"]["lcom_usd_per_kg_ch4"]
                                    / economics["economics_perfect"]["lcom_usd_per_kg_ch4"])
                        self.assertAlmostEqual(row["forecast_cost_ratio"], expected, places=8)
                        self.assertIsNotNone(row["outputs"]["economics_imperfect_with_faults"])
                    # A live showcase row must satisfy the saved-file contract, or
                    # results save successfully and then refuse to load back.
                    weather, _ = dashboard.collect_weather_cache(
                        [row.get("source_metadata", {}) for row in job["rows"]])
                    blob = dashboard.export_results("showcase", job["rows"], weather)
                    restored_kind, restored_rows, _ = dashboard.import_results(blob)
                    self.assertEqual(restored_kind, "showcase")
                    self.assertEqual([r["site"] for r in restored_rows],
                                     [r["site"] for r in job["rows"]])
                    dashboard.build_showcase_map(records=job["rows"])
                finally:
                    dashboard.RUN_JOBS.pop(job_id, None)

    def test_showcase_button_starts_background_batch_and_disables_while_running(self):
        import inspect
        application = dashboard.create_app()
        start_key = next(key for key in application.callback_map if key.startswith("..run-job.data"))
        start = application.callback_map[start_key]["callback"].__wrapped__
        arguments = {name: None for name in inspect.signature(start).parameters}
        locations = [{"site": "Paris", "lat": 48.5, "lon": 2.5},
                     {"site": "Madrid", "lat": 40.5, "lon": -3.5}]
        arguments.update(showcase_selection=locations, short_name="limping", long_method="battery",
                         information_mode="perfect_only", reactor_count=3, farm_mw=12,
                         saved_assumptions={"plant.electrolyser_kwh_per_kg_h2": 60})
        with patch.object(dashboard, "ctx", SimpleNamespace(triggered_id="run-showcase")), \
                patch.object(dashboard, "Thread") as thread:
            single_job, batch_job = start(**arguments)
            self.assertIs(single_job, dashboard.no_update)
            self.assertIs(thread.call_args.kwargs["target"], dashboard._execute_showcase_job)
            job_id, sites, parameters = thread.call_args.kwargs["args"]
            try:
                self.assertEqual(sites, locations)
                self.assertEqual(parameters["farm_mw"], 12)
                self.assertEqual(parameters["assumptions"]["plant.electrolyser_kwh_per_kg_h2"], 60)
                poll_key = next(key for key in application.callback_map
                                if key.startswith("..run-showcase.children"))
                poll = application.callback_map[poll_key]["callback"].__wrapped__
                response = poll(0, locations, batch_job, [])
                self.assertEqual(response[0], "Run 2 cases")
                self.assertTrue(response[1])
                arguments["showcase_job"] = batch_job
                self.assertEqual(start(**arguments), (dashboard.no_update, dashboard.no_update))
                self.assertEqual(thread.call_count, 1)
            finally:
                dashboard.RUN_JOBS.pop(job_id, None)

    def test_showcase_persists_between_batches_and_page_refreshes(self):
        application = dashboard.create_app()
        poll_key = next(key for key in application.callback_map if key.startswith("..run-showcase.children"))
        poll = application.callback_map[poll_key]["callback"].__wrapped__
        first = dashboard.load_showcase().iloc[0].to_dict()
        second = dashboard.load_showcase().iloc[1].to_dict()
        dashboard.SHOWCASE_RESULTS[first["site"]] = first
        self.assertEqual(poll(1, [], None, [])[4], [first])
        dashboard.SHOWCASE_RESULTS[second["site"]] = second
        self.assertEqual(poll(2, [], None, [first])[4], [first, second])
        self.assertEqual(poll(3, [], None, [])[4], [first, second])
        dashboard.SHOWCASE_RESULTS[first["site"]] = {**first, "lcom_usd_per_kg_ch4": 99}
        refreshed = poll(4, [], None, [first, second])[4]
        self.assertEqual(len(refreshed), 2)
        self.assertEqual(refreshed[0]["lcom_usd_per_kg_ch4"], 99)

    def test_air_hx_effectiveness_slider_reaches_the_plant(self):
        import inspect
        application = dashboard.create_app()
        start_key = next(key for key in application.callback_map
                         if key.startswith("..run-job.data"))
        start = application.callback_map[start_key]["callback"].__wrapped__
        arguments = {name: None for name in inspect.signature(start).parameters}
        arguments.update(lat=51.5, lon=-0.1, farm_mw=10, weather_source="demo",
                         short_name="limping", long_method="battery",
                         information_mode="perfect_only", reactor_count=1,
                         air_hx_effectiveness=0.9)
        with patch.object(dashboard, "ctx", SimpleNamespace(triggered_id="run-model")), \
                patch.object(dashboard, "Thread") as thread:
            start(**arguments)
            _, parameters = thread.call_args.kwargs["args"]
        self.assertEqual(parameters["air_hx_effectiveness"], 0.9)

        # The slider must override the saved assumption default, and omitting it
        # entirely must fall back to the dataclass default rather than failing.
        captured = []
        with patch.object(dashboard, "run_case",
                          side_effect=AssertionError("stop after plant build")) as run_case:
            for supplied, expected in ((0.9, 0.9), (None, dashboard.PlantParameters()
                                                    .air_exhaust_hx_effectiveness)):
                job_id = "test-hx"
                dashboard.RUN_JOBS[job_id] = {"status": "queued", "messages": []}
                try:
                    dashboard._execute_single_site_job(job_id, {
                        "lat": 51.5, "lon": -0.1, "farm_mw": 10,
                        "weather_source": "demo", "latest_year": 2025,
                        "information_mode": "perfect_only", "short_name": "limping",
                        "long_method": "battery", "f_ocp": 0.2, "f_socp": 1.0,
                        "reactor_count": 1, "reactor_scheduling_mode": "seasonal",
                        **({"air_hx_effectiveness": supplied} if supplied is not None else {}),
                    })
                finally:
                    dashboard.RUN_JOBS.pop(job_id, None)
                captured.append(run_case.call_args.kwargs["plant"].air_exhaust_hx_effectiveness)
                self.assertEqual(captured[-1], expected)

    def test_showcase_settings_summary_tracks_single_site_inputs(self):
        application = dashboard.create_app()
        entry = application.callback_map["showcase-settings.children"]
        self.assertEqual([item["id"] for item in entry["inputs"]],
                         [key for key, _ in dashboard.SHOWCASE_SETTING_FIELDS])
        callback = entry["callback"].__wrapped__
        values = [None] * len(dashboard.SHOWCASE_SETTING_FIELDS)
        values[0] = 12
        rows = callback(*values).children[1].children.children
        self.assertEqual(rows[0].children[1].children, "12")
        values[0] = 25
        rows = callback(*values).children[1].children.children
        self.assertEqual(rows[0].children[1].children, "25")

    def test_location_picker_uses_custom_coordinates_and_moves_marker(self):
        application = dashboard.create_app()
        # Exact key: the cached-site jump also writes lat/lon, as an allow_duplicate
        # output whose key carries a hash suffix, so a prefix match is ambiguous.
        pick_callback = application.callback_map["..lat.value...lon.value.."]["callback"]
        pick_runner = getattr(pick_callback, "__wrapped__", pick_callback)
        lat, lon = pick_runner({"points": [{"customdata": [48.5, 2.5]}]})
        self.assertEqual((lat, lon), (48.5, 2.5))

        marker_callback = application.callback_map["location-picker.figure"]["callback"]
        marker_runner = getattr(marker_callback, "__wrapped__", marker_callback)
        figure = marker_runner(lat, lon)
        self.assertEqual(float(figure.data[1].lat[0]), 48.5)
        self.assertEqual(float(figure.data[1].lon[0]), 2.5)
        self.assertGreater(len(figure.data[0].lat), 7000)

    def test_load_demo_button_restores_the_shipped_bundle(self):
        import tempfile
        from pathlib import Path
        application = dashboard.create_app()
        key = next(name for name in application.callback_map
                   if name.startswith("..results-download.data"))
        callback = application.callback_map[key]["callback"]
        runner = getattr(callback, "__wrapped__", callback)

        job_id = uuid4().hex
        dashboard.RUN_JOBS[job_id] = {"status": "queued", "messages": []}
        dashboard._execute_single_site_job(job_id, {
            "lat": 51.5, "lon": -0.1, "farm_mw": 10, "weather_source": "demo",
            "latest_year": 2025, "reactor_count": 2, "information_mode": "comparison",
            "run_scope": "specific", "short_name": "limping", "long_method": "battery",
            "f_ocp": 0.2, "f_socp": 1.0, "initial_soc_fraction": 0.5,
        })
        job = dashboard._job_snapshot(job_id)
        self.assertEqual(job["status"], "complete")
        single = dashboard.export_results(
            "single_site",
            {field: job[field] for field in dashboard.JOB_FIELDS if field in job}, [])
        location = dashboard.load_showcase().iloc[0].to_dict()
        showcase = dashboard.export_results("showcase", [location], [])
        dashboard.RUN_JOBS.pop(job_id, None)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            files = (folder / "showcase.json.gz", folder / "single_site.json.gz")
            # An absent bundle has to explain itself rather than fail silently, since
            # a fresh clone may not have run the build script yet.
            with patch.object(dashboard, "DEMO_FILES", files),                     patch.object(dashboard, "ctx",
                                 SimpleNamespace(triggered_id="load-demo")):
                _, no_job, site_status, map_status, message, _ = runner(
                    0, 0, 0, None, None, 1, 0, 0, None, None)
                self.assertIs(no_job, dashboard.no_update)
                self.assertIn("scripts/build_demo.py", message)

                files[0].write_bytes(showcase)
                files[1].write_bytes(single)
                dashboard.SHOWCASE_RESULTS.clear()
                _, new_job, site_status, map_status, message, _ = runner(
                    0, 0, 0, None, None, 2, 0, 0, None, None)

            # The demo bar carries its own status, so neither tab panel is disturbed.
            self.assertIs(site_status, dashboard.no_update)
            self.assertIs(map_status, dashboard.no_update)
            self.assertIn("Loaded 1 showcase location and", message)
            self.assertIn("1 single-site case,", message)
            self.assertEqual(list(dashboard.SHOWCASE_RESULTS), [location["site"]])
            restored = dashboard._job_snapshot(new_job["job_id"])
            try:
                self.assertEqual(restored["status"], "complete")
                self.assertEqual(list(restored["results"]), list(job["results"]))
                self.assertEqual(restored["information_mode"], "comparison")
            finally:
                dashboard.RUN_JOBS.pop(new_job["job_id"], None)

    def test_showcase_tab_button_loads_only_the_map_half_of_the_demo(self):
        import tempfile
        from pathlib import Path
        application = dashboard.create_app()
        key = next(name for name in application.callback_map
                   if name.startswith("..results-download.data"))
        callback = application.callback_map[key]["callback"]
        runner = getattr(callback, "__wrapped__", callback)
        location = dashboard.load_showcase().iloc[0].to_dict()

        with tempfile.TemporaryDirectory() as folder:
            blob = dashboard.export_results("showcase", [location], [])
            files = {}
            for variant, (_, description) in dashboard.DEMO_SHOWCASES.items():
                path = Path(folder) / f"showcase-{variant}.json.gz"
                path.write_bytes(blob)
                files[variant] = (path, description)
            # Each variant button loads its own file and names its configuration.
            for index, variant in enumerate(files):
                dashboard.SHOWCASE_RESULTS.clear()
                with patch.object(dashboard, "DEMO_SHOWCASES", files),                         patch.object(dashboard, "ctx", SimpleNamespace(
                            triggered_id=f"load-demo-showcase-{variant}")):
                    clicks = [0, 0]
                    clicks[index] = 1
                    (_, job, site_status, map_status,
                     demo_status, message) = runner(
                        0, 0, 0, None, None, 0, *clicks, None, None)

                # Loading the map must not disturb a single-site run on screen.
                self.assertIs(job, dashboard.no_update)
                self.assertIs(site_status, dashboard.no_update)
                self.assertIs(map_status, dashboard.no_update)
                self.assertIs(demo_status, dashboard.no_update)
                self.assertIn("Loaded 1 showcase location", message)
                self.assertIn(files[variant][1], message)
                self.assertEqual(list(dashboard.SHOWCASE_RESULTS), [location["site"]])

    def test_showcase_map_opens_framed_on_the_whole_latitude_range(self):
        import math
        figure = dashboard.build_showcase_map(
            records=dashboard.load_showcase().head(3).to_dict("records"), clickable=True)
        height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
        # Web Mercator: the visible latitude band follows from the plot height in
        # pixels, the zoom level and the centre.
        span = height * 2 * math.pi / (256 * 2 ** figure.layout.map.zoom)
        centre = math.log(math.tan(math.pi / 4
                                   + math.radians(figure.layout.map.center.lat) / 2))
        edge = lambda y: math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)
        top, bottom = edge(centre + span / 2), edge(centre - span / 2)
        # Tromso is the northernmost showcase site and Seville the southernmost;
        # both have to be on screen before the user pans.
        self.assertGreater(top, 69.65)
        self.assertLess(bottom, 37.39)

    def test_baseline_production_ratio_compares_like_with_like(self):
        from uuid import uuid4 as fresh
        job_id = fresh().hex
        dashboard.RUN_JOBS[job_id] = {"status": "queued", "messages": []}
        try:
            dashboard._execute_showcase_job(job_id, [{"site": "Paris", "lat": 48.5,
                                                      "lon": 2.5}], {
                "farm_mw": 10, "weather_source": "demo", "latest_year": 2025,
                "reactor_count": 2, "reactor_scheduling_mode": "seasonal",
                "information_mode": "comparison", "run_scope": "specific",
                "short_name": "limping", "long_method": "h2_co2",
                "f_ocp": 0.2, "f_socp": 1.0, "initial_soc_fraction": 0.5,
                "faults_enabled": True, "fault_seed": 7,
            })
            row = dashboard._job_snapshot(job_id)["rows"][0]
        finally:
            dashboard.RUN_JOBS.pop(job_id, None)

        baseline = row["outputs"]["economics_baseline"]
        self.assertIsNotNone(baseline)
        # Both sides carry the same faults, so the ratio measures storage and
        # scheduling alone rather than crediting them with equipment reliability.
        self.assertIn("same faults", baseline["baseline_methane_ratio_basis"])
        expanded = dashboard.expand_showcase_metrics(row)
        self.assertAlmostEqual(expanded["baseline_production_ratio"],
                               baseline["baseline_methane_ratio"], places=9)
        # Baseline over realised, so it reads the same way round as production_ratio
        # rather than inverted against it.
        self.assertAlmostEqual(
            expanded["baseline_production_ratio"],
            baseline["average_annual_methane_kg"] / expanded["ch4_faults"], places=9)
        # Storage and scheduling have to beat running flat out with neither, so the
        # do-nothing reference must land below what the plant realises.
        self.assertLess(expanded["baseline_production_ratio"], 1.0)
        # production_ratio is deliberately not asserted below one. Once the
        # forecast-driven cases run a plant sized on the training record, they are a
        # different plant from the perfect-information one and can carry more
        # throughput, so they can out-produce it in raw kilograms while costing more
        # capital to do it. LCOM is where that shows up, not production.
        self.assertGreater(expanded["production_ratio"], 0.0)
        self.assertIn("baseline_production_ratio", dashboard.SIMPLE_METRIC_META)
        self.assertIn("baseline_production_ratio",
                      [option["value"] for option in dashboard.metric_options(False)])

        # A record predating baselines must degrade to a blank metric, not an error.
        legacy = dict(row, outputs={key: value
                                    for key, value in row["outputs"].items()
                                    if key != "economics_baseline"})
        self.assertIsNone(
            dashboard.expand_showcase_metrics(legacy)["baseline_production_ratio"])
        dashboard.build_showcase_map("baseline_production_ratio",
                                     records=[row, legacy], clickable=True)

    def test_cards_and_sizing_follow_the_plant_the_category_actually_ran(self):
        import model
        weather = model.make_synthetic_weather("2014-01-01", years=4, latitude_deg=51.5)
        training, actual = model.split_weather_period(
            weather, training_years=2, evaluation_years=2)
        forecast = model.build_climatology_forecast(training, actual)
        result = model.run_case(
            actual, forecast, strategy=model.StrategyConfig(parallel_reactor_count=4),
            sizing_profile=training)
        self.assertIsNotNone(result.sizing_imperfect)
        self.assertNotEqual(result.sizing_imperfect.long_capacity_kwh,
                            result.sizing.long_capacity_kwh)

        # Reporting the perfect plant's capacities beside an imperfect result would
        # describe equipment that result never had.
        self.assertIs(dashboard._sizing_for_category(result, "perfect"), result.sizing)
        self.assertIs(dashboard._sizing_for_category(result, "imperfect"),
                      result.sizing_imperfect)
        self.assertIs(dashboard._sizing_for_category(result, "imperfect_with_faults"),
                      result.sizing_imperfect)
        self.assertIsNot(
            dashboard._equipment_register_for_category(result, "imperfect"),
            result.equipment_sizing)

        def long_energy(cards):
            card = next(str(c) for c in cards if "Long energy" in str(c))
            return card

        perfect_cards = dashboard._result_cards(result, "2016", "comparison",
                                                result_category="perfect")
        imperfect_cards = dashboard._result_cards(result, "2016", "comparison",
                                                  result_category="imperfect")
        self.assertNotEqual(long_energy(perfect_cards), long_energy(imperfect_cards))
        self.assertIsNotNone(dashboard.build_sizing_table(result, result.imperfect,
                                                          "imperfect"))

        # A run with one shared plant must keep reporting that one everywhere.
        shared = model.run_case(
            actual, forecast, strategy=model.StrategyConfig(parallel_reactor_count=4))
        self.assertIs(dashboard._sizing_for_category(shared, "imperfect"), shared.sizing)
        self.assertIs(dashboard._equipment_register_for_category(shared, "imperfect"),
                      shared.equipment_sizing)

    def test_map_hover_flags_a_site_whose_built_plant_runs_a_deficit(self):
        row = dashboard.load_showcase().iloc[0].to_dict()
        row["outputs"] = {"economics_perfect": {"lcom_usd_per_kg_ch4": 2},
                          "economics_imperfect": {"lcom_usd_per_kg_ch4": 3}}
        clean = dict(row, site="Sound Site", imperfect_deficit_mwh=0.0)
        short = dict(row, site="Short Site", imperfect_deficit_mwh=129.9)
        legacy = dict(row, site="Legacy Site")

        def note(record):
            return dashboard.expand_showcase_metrics(
                dashboard.normalize_showcase_ratio(record))["deficit_note"]

        self.assertEqual(note(short), dashboard.DEFICIT_NOTE)
        # A site that closes its cycle, and a record predating the split, say nothing.
        self.assertEqual(note(clean), "")
        self.assertEqual(note(legacy), "")

        figure = dashboard.build_showcase_map(records=[clean, short, legacy],
                                              clickable=True)
        results = next(trace for trace in figure.data
                       if getattr(trace, "name", None) != "Clickable locations")
        # The note is a sentence, not a statistic: bold, unlabelled, and last.
        self.assertNotIn("deficit_note=", results.hovertemplate)
        self.assertRegex(results.hovertemplate,
                         r"<b>%\{customdata\[\d+\]\}</b><extra>")

    def test_coordinate_grid_can_be_hidden_without_losing_the_result_dots(self):
        row = dashboard.load_showcase().iloc[0].to_dict()
        row["outputs"] = {"economics_perfect": {"lcom_usd_per_kg_ch4": 2},
                          "economics_imperfect": {"lcom_usd_per_kg_ch4": 3}}

        def grid_traces(figure):
            return [trace for trace in figure.data
                    if getattr(trace, "name", None) == "Clickable locations"]

        shown = dashboard.build_showcase_map(records=[row], clickable=True)
        hidden = dashboard.build_showcase_map(records=[row], clickable=True,
                                              show_grid=False)
        self.assertEqual(len(grid_traces(shown)), 1)
        self.assertEqual(grid_traces(hidden), [])
        # Hiding the grid must not take the results with it.
        self.assertEqual(len(hidden.data), len(shown.data) - 1)
        self.assertGreater(len(hidden.data), 0)

        # A result dot has to stay the bigger hover target, or the grid catches the
        # pointer first — which is the whole reason the toggle exists.
        results = next(trace for trace in shown.data
                       if getattr(trace, "name", None) != "Clickable locations")
        self.assertGreaterEqual(results.marker.sizemin, dashboard.MAP_GRID_SIZE)

        # Selection halos still have to sit beneath the result dots either way.
        for show_grid in (True, False):
            figure = dashboard.build_showcase_map(
                records=[row], clickable=True, show_grid=show_grid,
                selected=[{"site": "Paris", "lat": 48.5, "lon": 2.5}])
            names = [getattr(trace, "name", None) for trace in figure.data]
            self.assertLess(names.index("Selected for run"), len(names) - 1)

    def test_unvalidated_options_are_labelled_experimental(self):
        application = dashboard.create_app()
        layout = json.dumps(application.server.test_client()
                            .get("/_dash-layout").get_json())
        self.assertIn("experimental-badge", layout)
        # Each badge has to sit on the unvalidated option, not on the default the
        # results are quoted from.
        # Auto-calibration also carried a badge until it was withdrawn from the
        # dashboard; test_app_serves_layout_and_registers_callbacks asserts its absence.
        for flagged, default in (("Perfect daily switching", "Seasonal parallelisation"),):
            after = layout[layout.index(flagged) + len(flagged):]
            self.assertIn("EXPERIMENTAL", after[:600], flagged)
            self.assertNotIn(default, after[:after.index("EXPERIMENTAL")], default)

    def test_missing_ninja_key_warns_only_when_the_api_is_selected(self):
        application = dashboard.create_app()
        warn = application.callback_map["ninja-api-key-warning.children"]["callback"]
        runner = getattr(warn, "__wrapped__", warn)
        expected = ("A Renewables.ninja API key is required for loading external "
                    "weather data. Supply a key, or use synthetic weather data "
                    "(Offline synthetic) or pre-loaded profiles (Cached only).")

        # No key anywhere and the API selected: the only case that warns.
        with patch.object(dashboard, "_load_api_token", return_value=None):
            self.assertEqual(runner("ninja", None), expected)
            self.assertEqual(runner("ninja", "   "), expected)
            self.assertEqual(runner("ninja", "a-key"), "")
            for source in ("demo", "cached"):
                self.assertEqual(runner(source, None), "")
        # A key in .env is enough on its own, with the box left empty.
        with patch.object(dashboard, "_load_api_token", return_value="env-key"):
            self.assertEqual(runner("ninja", None), "")

        # The showcase tab mirrors the same warning rather than keeping its own copy.
        mirror = application.callback_map[
            "..showcase-ninja-usage.children...showcase-ninja-warning.children.."]
        mirror_runner = getattr(mirror["callback"], "__wrapped__", mirror["callback"])
        self.assertEqual(mirror_runner("usage", expected)[1], expected)

    def test_enabling_faults_switches_information_mode_rather_than_being_ignored(self):
        application = dashboard.create_app()
        runner = application.callback_map[
            "..information-mode.value...fault-mode-note.children.."
        ]["callback"].__wrapped__

        # run_case refuses faults without a forecast, so asking for faults has to ask
        # for the mode that can deliver them instead of being silently discarded.
        with patch.object(dashboard, "ctx",
                          SimpleNamespace(triggered_id="fault-enabled")):
            mode, note = runner("perfect_only", ["enabled"])
        self.assertEqual(mode, "comparison")
        self.assertIn("switched to Perfect vs imperfect", note)

        # The switch re-triggers the callback; its own explanation must survive that.
        with patch.object(dashboard, "ctx",
                          SimpleNamespace(triggered_id="information-mode")):
            mode, note = runner("comparison", ["enabled"])
        self.assertIs(mode, dashboard.no_update)
        self.assertIs(note, dashboard.no_update)

        # Choosing Perfect only deliberately is respected, but no longer silent.
        with patch.object(dashboard, "ctx",
                          SimpleNamespace(triggered_id="information-mode")):
            mode, note = runner("perfect_only", ["enabled"])
        self.assertIs(mode, dashboard.no_update)
        self.assertIn("will not be applied", note)

        with patch.object(dashboard, "ctx",
                          SimpleNamespace(triggered_id="fault-enabled")):
            self.assertEqual(runner("perfect_only", []), (dashboard.no_update, ""))

    def test_calibration_search_is_independent_of_the_slider_scale(self):
        import inspect
        application = dashboard.create_app()
        visibility = application.callback_map[
            "calibration-controls.style"]["callback"].__wrapped__
        self.assertEqual(visibility("auto")["display"], "grid")
        self.assertEqual(visibility("manual")["display"], "none")

        start_key = next(key for key in application.callback_map
                         if key.startswith("..run-job.data"))
        start = application.callback_map[start_key]["callback"].__wrapped__
        arguments = {name: None for name in inspect.signature(start).parameters}
        # The slider-scale inputs are no longer among the run states at all, so they
        # cannot reach the optimiser however they are set.
        self.assertNotIn("factor_minimum", arguments)
        self.assertIn("calibration_range", arguments)
        arguments.update(lat=51.5, lon=-0.1, farm_mw=10, weather_source="demo",
                         short_name="limping", long_method="battery",
                         information_mode="perfect_only", reactor_count=1,
                         calibration_range=[0.25, 1.5], calibration_increment=0.1)
        with patch.object(dashboard, "ctx",
                          SimpleNamespace(triggered_id="run-model")),                 patch.object(dashboard, "Thread") as thread:
            start(**arguments)
            _, parameters = thread.call_args.kwargs["args"]
        self.assertEqual(parameters["factor_minimum"], 0.25)
        self.assertEqual(parameters["factor_maximum"], 1.5)
        self.assertEqual(parameters["factor_increment"], 0.1)

    def test_demo_buttons_report_progress_while_they_load(self):
        application = dashboard.create_app()
        served = application.server.test_client().get("/_dash-dependencies").get_json()
        entry = next(item for item in served
                     if "results-download" in json.dumps(item.get("output")))
        # Loading a bundle takes seconds, so the click has to have a visible effect
        # before the work finishes, and the buttons must refuse a second click.
        spec = entry["running"]
        running, off = spec["running"], spec["runningOff"]
        for target in ("demo-progress.children", "showcase-demo-progress.children"):
            self.assertIn("Working", running[target])
            self.assertEqual(off[target], "")
        for button in ("load-demo", "load-demo-showcase-parallel",
                       "load-demo-showcase-single"):
            self.assertTrue(running[f"{button}.disabled"])
            self.assertFalse(off[f"{button}.disabled"])

    def test_the_storage_method_decides_whether_there_is_a_gate_buffer(self):
        import inspect
        import model
        application = dashboard.create_app()
        start_key = next(key for key in application.callback_map
                         if key.startswith("..run-job.data"))
        start = application.callback_map[start_key]["callback"].__wrapped__
        arguments = {name: None for name in inspect.signature(start).parameters}
        arguments.update(lat=51.5, lon=-0.1, farm_mw=10, weather_source="demo",
                         short_name="limping", information_mode="perfect_only",
                         reactor_count=1)

        # There is no switch any more: the vessel is what the methane method *is*.
        for long_method, expected in (("battery", False),
                                      (dashboard.PRODUCT_STORAGE_METHOD, True)):
            arguments["long_method"] = long_method
            with patch.object(dashboard, "ctx",
                              SimpleNamespace(triggered_id="run-model")),                     patch.object(dashboard, "Thread") as thread:
                start(**arguments)
                _, parameters = thread.call_args.kwargs["args"]
            self.assertEqual(parameters["long_method"], long_method)

        weather = model.make_synthetic_weather("2015-01-01", years=1,
                                               latitude_deg=51.5, seed=3)
        results = {}
        for product in (True, False):
            results[product] = model.run_case(
                weather, None, include_imperfect=False,
                strategy=model.StrategyConfig(parallel_reactor_count=4,
                                              product_storage=product),
            )
        buffered, floating = results[True], results[False]

        # With a vessel the gate flow is a single constant; without one it is whatever
        # was made that hour, and the vessel and its machinery cost nothing.
        self.assertAlmostEqual(
            buffered.perfect.hourly["methane_delivery_kg"].std(), 0.0, places=9)
        self.assertGreater(floating.perfect.hourly["methane_delivery_kg"].std(), 1.0)
        self.assertGreater(buffered.perfect.metadata["methane_storage"]["capacity_kg"], 0)
        self.assertEqual(floating.perfect.metadata["methane_storage"]["capacity_kg"], 0)
        self.assertGreater(buffered.economics_perfect["methane_storage_capex_usd"], 0)
        self.assertEqual(floating.economics_perfect["methane_storage_capex_usd"], 0)
        # Economics recomputes the profile, so it has to honour the method too rather
        # than reinstating a buffer the dispatch did not have.
        self.assertEqual(
            floating.perfect.metadata["methane_storage"]["expander_capacity_kw"], 0.0)
        # The buffered plant builds no upstream seasonal store: one store, not two.
        self.assertEqual(buffered.perfect.metadata["long_storage"]["capacity_kwh"], 0.0)
        self.assertGreater(floating.perfect.metadata["long_storage"]["capacity_kwh"], 0.0)


if __name__ == "__main__":
    unittest.main()
