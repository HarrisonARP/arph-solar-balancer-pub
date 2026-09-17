import unittest
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pandas as pd

import app as dashboard
import model
import pfd


class PfdTests(unittest.TestCase):
    def setUp(self):
        self.plant = model.PlantParameters()
        self.energy = model.calculate_plant_energy(self.plant)
        self.frame = pd.DataFrame({
            'methane_kg': [0., 2., 0., 2.], 'methane_delivery_kg': [1., 1., 1., 1.],
            'fresh_h2_kg': [0., .8, 0., .8], 'fresh_co2_kg': [0., 4., 0., 4.],
            'long_h2_charge_kg': [.2, 0., .2, 0.], 'long_co2_charge_kg': [1., 0., 1., 0.],
            'direct_h2_kg': [0., .2, 0., .2], 'direct_co2_kg': [0., 1., 0., 1.],
            'long_fuel_cell_h2_kg': [0., 0., 0., .4],
        }, index=pd.date_range('2020-01-01', periods=4, freq='h', tz='UTC'))
        self.sim = SimpleNamespace(hourly=self.frame.copy(), daily=pd.DataFrame(), metadata={
            'strategy': {'parallel_reactor_count': 4}, 'long_storage': {'method': 'h2_co2'}})
        self.ambient = pd.Series([280., 282., 284., 286.], index=self.frame.index)
        self.summary = pfd.summarize_streams(self.sim, self.plant, self.energy, self.ambient)

    def test_full_period_averages_and_stoichiometric_balances(self):
        streams = self.summary['streams']
        self.assertEqual(self.summary['hours'], 4)
        self.assertEqual(streams['S06']['flow_kg_h'], 1.)
        self.assertEqual(streams['S04']['flow_kg_h'], 2.)
        self.assertEqual(streams['S08']['flow_kg_h'], .1)
        self.assertEqual(streams['S09']['flow_kg_h'], .1)
        self.assertEqual(streams['S10']['flow_kg_h'], .1)
        self.assertEqual(streams['S11']['flow_kg_h'], .5)
        self.assertEqual(streams['S12']['flow_kg_h'], .5)
        self.assertAlmostEqual(streams['S02']['flow_kg_h'], .5 * self.plant.water_molar_mass_kg_per_mol / self.plant.hydrogen_molar_mass_kg_per_mol)
        self.assertAlmostEqual(streams['S03']['flow_kg_h'], self.energy.stoichiometry_kg_per_kg_ch4['H2'])
        self.assertAlmostEqual(streams['S05']['flow_kg_h'], 1 + streams['S07']['flow_kg_h'])
        air = (2.5 * self.plant.air_molar_mass_kg_per_mol
               / (self.plant.air_co2_mole_fraction
                  * self.plant.carbon_dioxide_molar_mass_kg_per_mol
                  * self.plant.dac_capture_efficiency))
        self.assertAlmostEqual(streams['S01']['flow_kg_h'], air)
        self.assertAlmostEqual(streams['S01']['temperature_c'], 283 - 273.15)
        self.assertTrue(streams['S03']['temperature_basis'].startswith('Assumed'))
        self.assertIsNone(streams['S06']['temperature_c'])

    def test_zero_and_missing_values_are_not_confused(self):
        self.sim.hourly.loc[:, :] = 0.
        zero = pfd.summarize_streams(self.sim, self.plant, self.energy)['streams']
        self.assertTrue(all(s['flow_kg_h'] == 0 for s in zero.values()))
        self.sim.hourly = self.sim.hourly.drop(columns='long_co2_charge_kg')
        missing = pfd.summarize_streams(self.sim, self.plant, self.energy)['streams']
        self.assertIsNone(missing['S11']['flow_kg_h'])
        self.assertIsNone(missing['S01']['flow_kg_h'])

    def test_summary_survives_compaction_and_render_does_not_modify_source(self):
        before = pfd.TEMPLATE.read_bytes()
        result = SimpleNamespace(perfect=self.sim, imperfect=None, energy=self.energy)
        dashboard._compact_dashboard_result(result, perfect_only=True, plant=self.plant, ambient_temperature=self.ambient)
        self.assertNotIn('fresh_h2_kg', self.sim.hourly)
        self.assertEqual(self.sim.metadata['pfd_summary']['streams']['S08']['flow_kg_h'], .1)
        doc = pfd.render_pfd(self.summary, 'Case <one> & two')
        root = ET.fromstring(doc[doc.index('<svg'):doc.index('</svg>') + 6])
        boxes = [n for n in root.iter() if (n.get('id') or '').startswith('data-S')]
        self.assertEqual(len(boxes), 12)
        for box in boxes:
            spans = box.findall(f'{{{pfd.SVG}}}tspan')
            self.assertEqual(len(spans), 4)
            self.assertIn('kg/h', spans[-1].text)
            self.assertNotIn('--', ''.join(n.text or '' for n in spans))
        self.assertIn('Case &lt;one&gt; &amp; two', doc)
        self.assertIn('4 SABATIER TRAINS INSTALLED', doc)
        self.assertEqual(before, pfd.TEMPLATE.read_bytes())

    def test_callback_uses_latest_job_selection_and_clears_stale_values(self):
        app = dashboard.create_app()
        callback = next(v['callback'].__wrapped__ for k, v in app.callback_map.items() if k.startswith('..pfd-frame'))
        self.sim.metadata['pfd_summary'] = self.summary
        second_summary = dict(self.summary, hours=8)
        second = SimpleNamespace(metadata={'pfd_summary': second_summary})
        result = SimpleNamespace(perfect=self.sim, economics_perfect={}, imperfect=None, economics_imperfect=None)
        other = SimpleNamespace(perfect=second, economics_perfect={}, imperfect=None, economics_imperfect=None)
        job = {'status': 'complete', 'results': {'limping|battery': result, 'limping|hydrogen': other},
               'information_mode': 'perfect_only', 'case_labels': {'limping|hydrogen': 'Hydrogen case'}}
        with patch.object(dashboard, '_job_snapshot', return_value=job):
            output = callback(1, {'job_id': 'new'}, 'limping|hydrogen', 'imperfect', None)
            self.assertIn('Hydrogen case', output[1])
            self.assertIn('8 hourly samples', output[1])
            self.assertIn('perfect', output[3])
            self.assertEqual(callback(2, {'job_id': 'new'}, 'limping|hydrogen', 'imperfect', output[3]), (dashboard.no_update,) * 4)
            job['status'] = 'running'
            pending = callback(3, {'job_id': 'new'}, 'limping|hydrogen', 'imperfect', output[3])
            self.assertIn('still calculating', pending[1])
            self.assertIn('Flow: --', pending[0])
            job['status'] = 'complete'
            job['information_mode'] = 'comparison'
            absent = callback(4, {'job_id': 'new'}, 'limping|hydrogen', 'imperfect_with_faults', None)
            self.assertIn('not calculated', absent[1])
        layout = app.server.test_client().get('/_dash-layout')
        self.assertEqual(layout.status_code, 200)
        self.assertIn('Process flow diagram', layout.get_data(as_text=True))
