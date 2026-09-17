import unittest
from uuid import uuid4

import app as dashboard


class InvestigationTests(unittest.TestCase):
    def test_one_factor_cases_keep_base_values_and_deduplicate_base(self):
        base = dict(short_name="limping", long_method="hydrogen", reactor_count=4,
                    reactor_scheduling_mode="seasonal")
        cases = dashboard.investigation_cases(dict(base, investigate=list(dashboard.INVESTIGATION_OPTIONS)))
        # One base case plus every one-factor variation: storage methods and operating
        # strategies less the base value, plus the train counts and schedule modes.
        expected = 1 + sum(len(options) - 1
                           for _, options in dashboard.INVESTIGATION_OPTIONS.values())
        self.assertEqual(len(cases), expected)
        self.assertEqual(cases[0]["label"], "Base case")
        for case in cases[1:]:
            self.assertEqual(sum(case[key] != value for key, value in base.items()), 1)
        self.assertEqual(len({tuple(case[key] for key in base) for case in cases}), expected)
        self.assertEqual(len(dashboard.investigation_cases(base)), 1)

    def test_investigation_runs_train_and_schedule_variants_and_plots_them(self):
        job_id = uuid4().hex
        dashboard.RUN_JOBS[job_id] = {"status": "queued", "messages": []}
        self.addCleanup(dashboard.RUN_JOBS.pop, job_id, None)
        dashboard._execute_single_site_job(job_id, {
            "lat": 51.5, "lon": -0.1, "farm_mw": 10, "weather_source": "demo",
            "latest_year": 2025, "reactor_count": 4, "reactor_scheduling_mode": "seasonal",
            "information_mode": "perfect_only", "run_scope": "investigate",
            "investigate": ["reactor_count", "reactor_scheduling_mode"],
            "short_name": "limping", "long_method": "battery", "f_ocp": 0.2,
            "f_socp": 1.0, "initial_soc_fraction": 0.35,
        })
        job = dashboard._job_snapshot(job_id)
        self.assertEqual(job["status"], "complete", job.get("error"))
        self.assertEqual(len(job["results"]), 7)
        self.assertEqual(len(job["case_labels"]), 7)
        counts = []
        for result in job["results"].values():
            counts.append(next(row["count"] for row in result.equipment_sizing
                               if row["unit_name"] == "Sabatier reactor"))
        self.assertEqual(sorted(counts), [1, 2, 3, 4, 4, 5, 6])
        figure = dashboard.build_case_comparison_figure(job, "total_capex_usd")
        self.assertEqual(len(figure.data[0].x), 7)
        self.assertTrue(all(value > 0 for value in figure.data[0].y))
        self.assertEqual(figure.data[0].name, "Perfect information")
        self.assertEqual(len(set(figure.data[0].x)), 7)
        application = dashboard.create_app()
        poll = next(value["callback"].__wrapped__ for key, value in application.callback_map.items()
                    if key.startswith("..run-status"))
        selected = list(job["results"])[1]
        output = poll(1, "limping", "battery", "manual", "perfect", {"job_id": job_id},
                      [["reactor_count"]], selected)
        self.assertFalse(output[3])
        self.assertFalse(output[4])
        output = poll(2, "limping", "battery", "manual", "perfect", {"job_id": job_id}, [], selected)
        self.assertTrue(output[3])

    def test_layout_and_dynamic_button(self):
        application = dashboard.create_app()
        layout = application.server.test_client().get("/_dash-layout").get_json()
        def nodes(node):
            if isinstance(node, dict):
                yield node
                for value in node.values():
                    yield from nodes(value)
            elif isinstance(node, list):
                for value in node:
                    yield from nodes(value)
        all_nodes = list(nodes(layout))
        toggles = [node for node in all_nodes if isinstance(node.get("props", {}).get("id"), dict)
                   and node["props"]["id"].get("type") == "investigate"]
        self.assertEqual(len(toggles), 4)
        button = application.callback_map["run-all.children"]["callback"].__wrapped__
        self.assertEqual(button([[], [], [], []]), "Investigate 0 selected parameters")
        self.assertEqual(button([["short_name"], [], [], []]), "Investigate 1 selected parameter")
        self.assertEqual(button([["short_name"], ["long_method"], [], []]), "Investigate 2 selected parameters")
        self.assertTrue(any(node.get("type") == "Details" for node in all_nodes))
        icons = [node["props"] for node in all_nodes if node.get("props", {}).get("className") == "info-icon"]
        self.assertTrue(icons)
        self.assertTrue(all(icon["title"] and icon["tabIndex"] == 0 for icon in icons))
