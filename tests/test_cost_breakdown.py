import unittest
from types import SimpleNamespace

import pandas as pd

import app as dashboard
import model


class CostBreakdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energy = model.calculate_plant_energy()

    def economics(self, method, production=8760):
        hourly = pd.DataFrame({"methane_kg": [0., 2., 0., 2.]},
                              index=pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC"))
        simulation = model.SimulationResult(hourly, hourly.copy(),
                                            {"average_annual_methane_kg": production})
        short = model.StorageParameters(capacity_kwh=100, max_charge_kw=10, max_discharge_kw=20)
        long = model.StorageParameters(method=method, capacity_kwh=3333,
                                       max_charge_kw=100, max_discharge_kw=200,
                                       co2_capacity_kg=200, hydrogen_charge_capacity_kw=100,
                                       co2_charge_compressor_capacity_kw=20)
        return model.calculate_economics(simulation, self.energy, 100, short, long)

    def test_breakdowns_reconcile_for_every_storage_method(self):
        for method in dashboard.LONG_STORAGE_METHODS:
            with self.subTest(method=method):
                economics = self.economics(method)
                capital = economics["capex_breakdown_usd"]
                operating = economics["opex_breakdown_usd_per_year"]
                self.assertAlmostEqual(sum(capital.values()), economics["total_capex_usd"])
                self.assertAlmostEqual(sum(operating.values()), economics["annual_opex_usd_per_year"])
                self.assertAlmostEqual(capital["Short-term storage"] + capital["Long-term storage"],
                                       economics["storage_capex_usd"])
                self.assertAlmostEqual(capital["Product methane buffer"], economics["methane_storage_capex_usd"])
                self.assertNotIn("Storage Fuel Cell", capital)
                self.assertNotIn("Storage Co2 Compressor", capital)
                self.assertAlmostEqual(operating["Production (variable)"], 8760 * 0.05)

    def test_charts_have_one_reconciling_stack_per_case_and_support_filters(self):
        results = {}
        for method in dashboard.LONG_STORAGE_METHODS:
            perfect = self.economics(method)
            imperfect = self.economics(method, production=4000)
            results[f"limping|{method}"] = SimpleNamespace(
                perfect=object(), imperfect=object(), economics_perfect=perfect,
                economics_imperfect=imperfect, imperfect_with_faults=None)
        job = {"results": results, "information_mode": "comparison"}
        for category, attribute in (("perfect", "economics_perfect"), ("imperfect", "economics_imperfect")):
            figures = dashboard.build_cost_breakdown_figures(job, category)
            for figure, total in zip(figures, ("total_capex_usd", "annual_opex_usd_per_year")):
                self.assertEqual(figure.layout.barmode, "stack")
                # One stack per case, and the cases are built from every storage
                # method above, so derive it rather than pin the count.
                self.assertEqual(len(figure.data[0].x),
                                 len(dashboard.LONG_STORAGE_METHODS))
                for index, result in enumerate(results.values()):
                    self.assertAlmostEqual(sum(trace.y[index] for trace in figure.data),
                                           getattr(result, attribute)[total])
        single = dashboard.build_cost_breakdown_figures(job, selected_cases=["limping|battery"])
        self.assertEqual(len(single[0].data[0].x), 1)
        empty = dashboard.build_cost_breakdown_figures(job, selected_cases=[])
        self.assertEqual(len(empty[0].data), 0)
        no_faults = dashboard.build_cost_breakdown_figures(job, "imperfect_with_faults")
        self.assertEqual(len(no_faults[0].data), 0)
        job["information_mode"] = "perfect_only"
        perfect_only = dashboard.build_cost_breakdown_figures(job)
        self.assertIn("perfect information", perfect_only[0].layout.title.text)

    def test_cost_callback_renders_after_completion_and_caches(self):
        application = dashboard.create_app()
        callback = next(value["callback"].__wrapped__ for key, value in application.callback_map.items()
                        if key.startswith("..capex-breakdown"))
        economics = self.economics("hydrogen")
        result = SimpleNamespace(perfect=object(), economics_perfect=economics)
        dashboard.RUN_JOBS["cost-test"] = {"status": "complete", "messages": [],
                                         "results": {"limping|hydrogen": result},
                                         "information_mode": "perfect_only"}
        self.addCleanup(dashboard.RUN_JOBS.pop, "cost-test", None)
        args = ({"job_id": "cost-test"}, "imperfect", ["limping|hydrogen"], ["hydrogen"], ["limping"])
        figures = callback(1, *args)
        self.assertGreater(len(figures[0].data), 0)
        self.assertEqual(callback(2, *args), (dashboard.no_update, dashboard.no_update))
