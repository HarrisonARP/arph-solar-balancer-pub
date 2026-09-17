import base64
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import app
import model
import result_files as files


class ResultFileTests(unittest.TestCase):
    def test_showcase_ratio_combines_weather_and_fault_penalties(self):
        row = app.load_showcase().iloc[0].to_dict()
        row["outputs"] = {
            "economics_perfect": {"lcom_usd_per_kg_ch4": 2},
            "economics_imperfect": {"lcom_usd_per_kg_ch4": 3},
            "economics_imperfect_with_faults": {"lcom_usd_per_kg_ch4": 5},
        }
        self.assertEqual(files.normalize_showcase_ratio(row)["forecast_cost_ratio"], 2.5)
        _, loaded, _ = files.import_results(files.export_results("showcase", [row]))
        self.assertEqual(loaded[0]["forecast_cost_ratio"], 2.5)
        self.assertEqual(app.showcase_data([row]).iloc[0].forecast_cost_ratio, 2.5)
        row["outputs"]["economics_imperfect_with_faults"] = None
        self.assertEqual(files.normalize_showcase_ratio(row)["forecast_cost_ratio"], 1.5)
        row["outputs"]["economics_imperfect"] = None
        self.assertIsNone(files.normalize_showcase_ratio(row)["forecast_cost_ratio"])
        app.build_showcase_map("forecast_cost_ratio", records=[row])
        row["outputs"]["economics_perfect"]["lcom_usd_per_kg_ch4"] = 0
        self.assertIsNone(files.normalize_showcase_ratio(row)["forecast_cost_ratio"])

    def test_single_site_roundtrip_restores_plots_sizing_and_pfd(self):
        weather = model.make_synthetic_weather("2020-01-01", years=1).iloc[:24 * 10]
        result = model.run_case(weather, weather, strategy=model.StrategyConfig(parallel_reactor_count=2))
        app._compact_dashboard_result(result, perfect_only=False, plant=model.PlantParameters(),
                                      ambient_temperature=weather.ambient_temperature_k)
        job = {"results": {"limping|battery": result}, "information_mode": "comparison",
               "source_metadata": {"source": "synthetic_demo"}, "evaluation_period": "2020",
               "inputs": {"ninja_api_key": "test-secret", "farm_mw": 10}}
        raw = files.export_results("single_site", job)
        self.assertNotIn(b"test-secret", gzip.decompress(raw))
        kind, loaded, cache = files.import_results(raw)
        self.assertEqual(kind, "single_site")
        self.assertEqual(cache, [])
        restored = loaded["results"]["limping|battery"]
        pd.testing.assert_frame_equal(restored.perfect.hourly, result.perfect.hourly, check_freq=False)
        self.assertEqual(restored.economics_perfect, result.economics_perfect)
        self.assertEqual(restored.perfect.metadata["pfd_summary"], result.perfect.metadata["pfd_summary"])
        self.assertEqual(len(restored.equipment_sizing), len(result.equipment_sizing))
        self.assertGreater(len(model.build_dispatch_figure(restored.perfect).data), 0)
        self.assertIsNotNone(app.build_sizing_table(restored))
        application = app.create_app()
        callback_key = next(key for key in application.callback_map if key.startswith("..results-download.data"))
        callback = application.callback_map[callback_key]["callback"].__wrapped__
        upload = "data:application/gzip;base64," + base64.b64encode(raw).decode()
        with patch.object(app, "ctx", SimpleNamespace(triggered_id="load-site-results")):
            response = callback(0, 0, 0, upload, None, 0, 0, 0, None, None)
        self.assertIn("Loaded 1 cases", response[2])
        restored_job_id = response[1]["job_id"]
        try:
            restored_job = app._job_snapshot(restored_job_id)
            self.assertEqual(restored_job["status"], "complete")
            restored_case = restored_job["results"]["limping|battery"]
            self.assertEqual(restored_case.case_id, result.case_id)
            self.assertGreater(len(app._result_cards(restored_case, "2020", "comparison")), 0)
        finally:
            app.RUN_JOBS.pop(restored_job_id, None)

    def test_columnar_frames_preserve_awkward_values(self):
        index = pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC",
                              name="timestamp").as_unit("ns")
        frame = pd.DataFrame({
            "floats": [float(value) * 1.5 for value in range(48)],
            "gaps": [float("nan") if value % 7 == 0 else float(value) for value in range(48)],
            "unbounded": [float("inf") if value == 3 else
                          (float("-inf") if value == 4 else float(value)) for value in range(48)],
            "counts": list(range(48)),
            "flags": [value % 2 == 0 for value in range(48)],
            "labels": [f"state_{value % 3}" for value in range(48)],
        }, index=index)
        # NaN and the two infinities have to stay distinguishable: allow_nan=False
        # rules out bare tokens, and a null alone cannot tell a gap from an overflow.
        pd.testing.assert_frame_equal(files._decode(files._encode(frame)), frame,
                                      check_freq=False)
        for label, awkward in (("irregular", frame.iloc[[0, 1, 5, 20, 47]]),
                               ("single row", frame.iloc[:1]),
                               ("no rows", frame.iloc[:0]),
                               ("tz-naive", frame.tz_localize(None)),
                               ("empty", pd.DataFrame())):
            with self.subTest(label):
                pd.testing.assert_frame_equal(files._decode(files._encode(awkward)),
                                              awkward, check_freq=False)
        rounded = files._decode(files._encode(frame[["floats"]], 1))
        pd.testing.assert_frame_equal(rounded, frame[["floats"]].round(1), check_freq=False)

    def test_version_one_files_still_load(self):
        row = app.load_showcase().iloc[0].to_dict()
        current = json.loads(gzip.decompress(files.export_results("showcase", [row])))
        self.assertEqual(current["version"], 2)

        # A file written before the columnar format must keep loading, so a saved run
        # does not become unreadable when the format moves on.
        frame = pd.DataFrame({"a": [1.0, 2.0]},
                             index=pd.date_range("2020-01-01", periods=2, freq="h",
                                                 tz="UTC").as_unit("ns"))
        legacy = {"_type": "DataFrame",
                  "table": frame.to_json(orient="table", date_format="iso")}
        pd.testing.assert_frame_equal(files._decode(legacy), frame, check_freq=False)
        document = dict(current, version=1)
        kind, payload, _ = files.import_results(
            gzip.compress(json.dumps(document).encode("utf-8")))
        self.assertEqual((kind, payload[0]["site"]), ("showcase", row["site"]))
        with self.assertRaises(ValueError):
            files.import_results(gzip.compress(
                json.dumps(dict(current, version=99)).encode("utf-8")))

    def test_demo_bundles_keep_only_what_the_dashboard_reads_back(self):
        weather = model.make_synthetic_weather("2020-01-01", years=1).iloc[:24 * 10]
        result = model.run_case(weather, weather,
                                strategy=model.StrategyConfig(parallel_reactor_count=2))
        app._compact_dashboard_result(result, perfect_only=False, plant=model.PlantParameters(),
                                      ambient_temperature=weather.ambient_temperature_k)
        job = {"results": {"limping|battery": result}, "information_mode": "comparison",
               "source_metadata": {"source": "synthetic_demo"}, "evaluation_period": "2020"}
        full = files.export_results("single_site", job)
        trimmed = files.export_results("single_site", app.compact_for_demo(dict(job)),
                                       decimals=6)
        self.assertLess(len(gzip.decompress(trimmed)), len(gzip.decompress(full)))

        _, loaded, _ = files.import_results(trimmed)
        restored = loaded["results"]["limping|battery"]
        # Everything the plots and tables read must survive the trim.
        for column in app.DISPLAY_HOURLY_COLUMNS:
            self.assertIn(column, restored.perfect.hourly.columns, column)
        self.assertIn("pfd_summary", restored.perfect.metadata)
        self.assertGreater(len(model.build_dispatch_figure(restored.perfect).data), 0)
        self.assertGreater(len(model.build_reactor_count_figure(restored.perfect).data), 0)
        self.assertIsNotNone(app.build_limiting_subsystem_figure(restored.perfect))
        self.assertIsNotNone(app.build_active_reactor_figure(restored.perfect))
        self.assertIsNotNone(app.build_sizing_table(restored))
        self.assertGreater(len(app._result_cards(restored, "2020", "comparison")), 0)

    def test_save_size_is_estimated_before_the_limit_is_hit(self):
        hours = pd.date_range("2020-01-01", periods=8760, freq="h", tz="UTC")
        simulation = SimpleNamespace(
            hourly=pd.DataFrame(0.0, index=hours, columns=list("abcdefghij")),
            daily=pd.DataFrame())
        case = SimpleNamespace(perfect=simulation, imperfect=None,
                               imperfect_with_faults=None, baseline=None)
        payload = {"results": {"one": case}}
        self.assertEqual(files.estimate_export_bytes(payload),
                         8760 * 10 * files.BYTES_PER_STORED_VALUE)
        self.assertIn("MB", files.describe_export_size(payload))
        self.assertIsNone(files.describe_export_size({"results": {}}))

        huge = pd.DataFrame(0.0, index=hours, columns=[str(n) for n in range(4000)])
        over = {"results": {"one": SimpleNamespace(
            perfect=SimpleNamespace(hourly=huge, daily=pd.DataFrame()),
            imperfect=None, imperfect_with_faults=None, baseline=None)}}
        self.assertGreater(files.estimate_export_bytes(over), files.MAX_FILE_BYTES)
        self.assertIn("over the", files.describe_export_size(over))
        # The failure itself has to say what went wrong, not just that it did.
        self.assertIn("Shorten the evaluation period",
                      files.describe_export_overflow(files.MAX_FILE_BYTES * 2))

    def test_cached_weather_roundtrip_can_run_without_api(self):
        config = model.WeatherConfig(51.5, -0.1, latest_year=2020, training_years=0, evaluation_years=1)
        frame = model.make_synthetic_weather("2020-01-01", years=1)
        row = app.load_showcase().iloc[0].to_dict()
        key = model._weather_cache_key(config)
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            csv, metadata = model._cache_paths(config, 2020, source)
            csv.parent.mkdir(parents=True)
            frame.rename_axis("timestamp").to_csv(csv)
            metadata.write_text(json.dumps({"test": "weather metadata"}), encoding="utf-8")
            entries, missing = files.collect_weather_cache(
                [{"cache_key": key, "years": [2020]}] * 2, source)
            self.assertEqual(len(entries), 1)
            self.assertEqual(missing, [])

            # A site whose cached years have been deleted is reported by name and
            # does not prevent the remaining results from being exported.
            entries, missing = files.collect_weather_cache([
                {"cache_key": key, "years": [2020]},
                {"cache_key": "0" * 16, "years": [2020, 2021], "site": "near Warsaw"},
            ], source)
            self.assertEqual(len(entries), 1)
            self.assertEqual(missing, ["near Warsaw (2 years)"])
            self.assertTrue(files.export_results("showcase", [row], entries))
            kind, payload, restored_cache = files.import_results(files.export_results("showcase", [row], entries))
            self.assertEqual(kind, "showcase")
            self.assertEqual(payload[0]["site"], row["site"])
            with self.assertRaises(ValueError):
                files.restore_weather_cache(restored_cache + [{"cache_key": "invalid", "year": 2020}], target)
            self.assertEqual(list(Path(target).iterdir()), [])
            self.assertEqual(files.restore_weather_cache(restored_cache, target), 1)
            with patch.object(model, "_load_api_token", return_value=None), \
                    patch.object(model.requests.Session, "get") as get:
                loaded, _ = model.fetch_solar_profile(config, cache_dir=target)
                get.assert_not_called()
            pd.testing.assert_frame_equal(loaded, frame, check_freq=False)
            self.assertEqual(json.loads(model._cache_paths(config, 2020, target)[1].read_text()),
                             {"test": "weather metadata"})

            # Restoring a year that is already cached must leave the file untouched,
            # so loading results repeatedly cannot churn a tracked weather cache.
            cached_csv = model._cache_paths(config, 2020, target)[0]
            cached_csv.write_text("sentinel", encoding="utf-8")
            self.assertEqual(files.restore_weather_cache(restored_cache, target), 1)
            self.assertEqual(cached_csv.read_text(encoding="utf-8"), "sentinel")

    def test_invalid_cache_and_version_are_rejected(self):
        row = app.load_showcase().iloc[0].to_dict()
        with tempfile.TemporaryDirectory() as target:
            with self.assertRaises(ValueError):
                files.restore_weather_cache([{"cache_key": "../../escape", "year": 2020}], target)
            self.assertEqual(list(Path(target).iterdir()), [])
        document = json.loads(gzip.decompress(files.export_results("showcase", [row])))
        document["version"] = 999
        with self.assertRaisesRegex(ValueError, "version"):
            files.import_results(json.dumps(document).encode())
        row["lat"] = 999
        with self.assertRaisesRegex(ValueError, "coordinates"):
            files.import_results(files.export_results("showcase", [row]))

    def test_showcase_download_load_merges_and_reports_bad_upload(self):
        application = app.create_app()
        callback_key = next(key for key in application.callback_map if key.startswith("..results-download.data"))
        callback = application.callback_map[callback_key]["callback"].__wrapped__
        first, second = [app.load_showcase().iloc[i].to_dict() for i in (0, 1)]
        with patch.dict(app.SHOWCASE_RESULTS, {first["site"]: first}, clear=True):
            with patch.object(app, "ctx", SimpleNamespace(triggered_id="save-showcase-results")):
                response = callback(0, 1, 0, None, None, 0, 0, 0, None, None)
            download = response[0]
            self.assertTrue(download["base64"])
            upload = "data:application/gzip;base64," + download["content"]
            app.SHOWCASE_RESULTS.clear()
            app.SHOWCASE_RESULTS[second["site"]] = second
            with patch.object(app, "ctx", SimpleNamespace(triggered_id="load-showcase-results")):
                response = callback(0, 1, 0, None, upload, 0, 0, 0, None, None)
                self.assertIn("Loaded 1 showcase", response[3])
                self.assertEqual(set(app.SHOWCASE_RESULTS), {first["site"], second["site"]})
                response = callback(0, 1, 0, None, "data:application/json;base64,bad!", 0, 0, 0, None, None)
                self.assertIn("Could not", response[3])
                self.assertEqual(len(app.SHOWCASE_RESULTS), 2)

    def test_showcase_legend_is_below_map_and_colour_scale_outside(self):
        figure = app.build_showcase_map(records=app.load_showcase().head(1).to_dict("records"), clickable=True)
        self.assertLess(figure.layout.legend.y, 0)
        self.assertGreater(figure.layout.coloraxis.colorbar.x, 1)
