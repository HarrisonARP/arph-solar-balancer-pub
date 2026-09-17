"""Stream summaries and SVG annotation for the reviewed Solar Balancer PFD."""
from functools import lru_cache
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import pandas as pd

from functions import dry_air_mass_for_co2_kg

TEMPLATE = Path(__file__).resolve().parent / 'docs' / 'pfd' / 'main-process.svg'
SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
NAMES = ('Air', 'Water', 'H2', 'CO2', 'Outlet', 'CH4', 'Water*',
         'H2 in', 'H2 feed', 'FC H2', 'CO2 in', 'CO2 feed')


def summarize_streams(simulation, plant, energy, ambient_temperature=None):
    """Average hourly kg increments over every simulated hour, including downtime.

    The model uses fixed one-hour timesteps. Retain this small summary before
    dashboard compaction removes the fresh-feed and storage-transfer columns.
    """
    frame = simulation.hourly

    def mean(column):
        if column not in frame or frame.empty:
            return None
        values = pd.to_numeric(frame[column], errors='coerce')
        if not values.map(lambda v: pd.notna(v) and math.isfinite(v)).all():
            return None
        return float(values.sum() / len(frame))

    def add(*values):
        return sum(values) if all(v is not None for v in values) else None

    def scale(value, factor):
        return value * factor if value is not None else None

    methane = mean('methane_kg')
    fresh_h2, fresh_co2 = mean('fresh_h2_kg'), mean('fresh_co2_kg')
    h2_charge, co2_charge = mean('long_h2_charge_kg'), mean('long_co2_charge_kg')
    captured_co2 = add(fresh_co2, co2_charge)
    # Share the energy model's air-mass basis so the diagram cannot drift from E_req.
    air_per_co2 = dry_air_mass_for_co2_kg(
        1.0, plant.air_co2_mole_fraction, plant.air_molar_mass_kg_per_mol,
        plant.carbon_dioxide_molar_mass_kg_per_mol, plant.dac_capture_efficiency)
    ratios = energy.stoichiometry_kg_per_kg_ch4
    flows = [
        scale(captured_co2, air_per_co2),
        scale(add(fresh_h2, h2_charge), plant.water_molar_mass_kg_per_mol / plant.hydrogen_molar_mass_kg_per_mol),
        scale(methane, ratios['H2']), fresh_co2,
        scale(methane, 1 + ratios['H2O_product']), mean('methane_delivery_kg'),
        scale(methane, ratios['H2O_product']), h2_charge,
        mean('direct_h2_kg'), mean('long_fuel_cell_h2_kg'), co2_charge, mean('direct_co2_kg'),
    ]
    notes = [
        'Calculated air intake for fresh process CO2 plus CO2 storage charging, using the model air composition and single-pass DAC capture efficiency.',
        'Stoichiometric electrolyser water consumption for fresh H2 plus H2 storage charging; excludes any water recycle or excess feed.',
        'Total H2 feed to methanation, including stored H2 rejoining the header; calculated from methane production.',
        'Fresh CO2 through the process compressor to methanation. Stored CO2 rejoins downstream; storage charging is S11.',
        'Total hot reactor product mass: methane plus stoichiometric water, assuming complete conversion and no slip.',
        'Continuous methane delivery after the product buffer, from the simulated delivery series.',
        'Stoichiometric reaction-water production; actual separation recovery is not modelled.',
        'Gross H2 storage charging, not net inventory change over the simulation.',
        'Stored H2 discharged directly to the methanation feed.',
        'Stored H2 consumed by the fuel cell.',
        'Gross CO2 storage charging, not net inventory change over the simulation.',
        'Stored CO2 discharged directly to the methanation feed.',
    ]
    streams = {}
    for index, (name, flow, note) in enumerate(zip(NAMES, flows, notes), 1):
        streams[f'S{index:02}'] = dict(name=name, flow_kg_h=flow,
                                     temperature_c=None, pressure_bar=None,
                                     temperature_basis='Not modelled', pressure_basis='Not modelled', note=note)
    if ambient_temperature is not None and not frame.empty:
        ambient = pd.to_numeric(ambient_temperature.reindex(frame.index), errors='coerce')
        valid = ambient.map(lambda v: pd.notna(v) and math.isfinite(v))
        if valid.all():
            streams['S01'].update(temperature_c=float(ambient.mean()) - 273.15,
                                  temperature_basis='Time average of weather input')
        else:
            streams['S01'].update(temperature_c=plant.reference_ambient_temperature_k - 273.15,
                                  temperature_basis='Assumed reference ambient; weather temperature incomplete')
    else:
        streams['S01'].update(temperature_c=plant.reference_ambient_temperature_k - 273.15,
                              temperature_basis='Assumed reference ambient')
    streams['S01'].update(pressure_bar=plant.air_pressure_bar, pressure_basis='Assumed air pressure')
    for tag in ('S03', 'S04'):
        streams[tag].update(temperature_c=plant.intercool_temperature_k - 273.15,
                            temperature_basis='Assumed model feed-conditioning temperature',
                            pressure_bar=plant.co2_outlet_pressure_bar,
                            pressure_basis='Assumed process pressure; pressure drops not modelled')
    streams['S05'].update(temperature_c=plant.sabatier_temperature_k - 273.15,
                          temperature_basis='Assumed reactor setpoint proxy; effluent temperature not simulated',
                          pressure_bar=plant.co2_outlet_pressure_bar,
                          pressure_basis='Assumed process pressure; pressure drops not modelled')
    # The product buffer is built only by the methane long-storage method. Without it
    # S06 is production, not a constant delivery, and there is no expansion to name as
    # the pressure basis. A zero vessel capacity is what distinguishes the two.
    buffered = bool((simulation.metadata.get('methane_storage') or {}).get('capacity_kg'))
    streams['S06'].update(
        pressure_bar=plant.methane_delivery_pressure_bar,
        pressure_basis=('Assumed methane delivery pressure downstream of storage expansion'
                        if buffered else 'Assumed methane delivery pressure; no product buffer built'),
        note=('Continuous methane delivery after the product buffer, from the simulated '
              'delivery series.' if buffered else
              'Methane leaving the gate as it is made: this storage method builds no '
              'product buffer, so delivery follows production hour by hour.'),
    )
    streams['S08'].update(
        pressure_bar=plant.hydrogen_storage_pressure_bar,
        pressure_basis='Assumed hydrogen vessel pressure downstream of storage compression',
    )
    for tag in ('S09', 'S10'):
        streams[tag].update(
            pressure_bar=plant.co2_outlet_pressure_bar,
            pressure_basis='Assumed process pressure downstream of hydrogen storage expansion',
        )
    streams['S11'].update(
        pressure_bar=plant.co2_storage_pressure_bar,
        pressure_basis='Assumed CO2 vessel pressure downstream of storage compression',
    )
    streams['S12'].update(
        pressure_bar=plant.co2_outlet_pressure_bar,
        pressure_basis='Assumed process pressure downstream of CO2 storage expansion',
    )
    strategy = simulation.metadata.get('strategy', {})
    return dict(streams=streams, hours=len(frame),
                start=str(frame.index[0]) if len(frame) else '',
                end=str(frame.index[-1]) if len(frame) else '',
                reactor_count=strategy.get('parallel_reactor_count'),
                storage_method=simulation.metadata.get('long_storage', {}).get('method'))


def format_number(value):
    return 'N/A' if value is None or not math.isfinite(value) else f'{value:,.3g}'


@lru_cache(maxsize=2)
def _template(modified):
    return TEMPLATE.read_text(encoding='utf-8')


def render_pfd(summary=None, message='Run a case to populate the stream labels.'):
    """Annotate a fresh copy of the reviewed SVG; never alter the editable source."""
    root = ET.fromstring(_template(TEMPLATE.stat().st_mtime_ns))
    nodes = {node.get('id'): node for node in root.iter() if node.get('id')}

    def replace_text(id, lines, compact=False):
        element = nodes.get(id)
        if element is None:
            return
        spans = element.findall(f'{{{SVG}}}tspan')
        if not spans:
            return
        x, y = spans[0].get('x'), float(spans[0].get('y'))
        for child in list(element):
            element.remove(child)
        if compact:
            element.set('font-size', '12')
            y -= 7
        for index, line in enumerate(lines):
            child = ET.SubElement(element, f'{{{SVG}}}tspan', x=x, y=str(y + index * (16 if compact else 20)))
            child.text = line

    replace_text('subtitle', [message])
    replace_text('footer', [
        'Flows: kg/h, averaged over ALL simulation hours including shutdowns. * = assumed condition. N/A = not modelled / unavailable.',
        'Blue = material | Gold = electricity | Olive = solids | Stream details explain calculation bases. Illustrative toy model, ARPH 2026.',
    ])
    if summary:
        for tag, stream in summary['streams'].items():
            temp, pressure = stream['temperature_c'], stream['pressure_bar']
            temp_text = 'N/A' if temp is None else f'{format_number(temp)} °C'
            pressure_text = 'N/A' if pressure is None else f'{format_number(pressure)} bar'
            if temp is not None and stream['temperature_basis'].startswith('Assumed'):
                temp_text += ' *'
            if pressure is not None and stream['pressure_basis'].startswith('Assumed'):
                pressure_text += ' *'
            replace_text('data-' + tag, [f'{tag} {stream["name"]}', f'T: {temp_text}',
                                        f'P: {pressure_text}', f'{format_number(stream["flow_kg_h"])} kg/h'], compact=True)
            tooltip = ET.SubElement(nodes['data-' + tag], f'{{{SVG}}}title')
            tooltip.text = f'{stream["note"]} Temperature: {stream["temperature_basis"]}. Pressure: {stream["pressure_basis"]}.'
        count = summary.get('reactor_count')
        if count:
            replace_text('parallel-caption', [f'{count} SABATIER TRAIN{"S" if count != 1 else ""} INSTALLED',
                                              'Symbols show train topology; stream flows are totals across all trains.'])
    svg = ET.tostring(root, encoding='unicode')
    return ('<!doctype html><html><head><meta charset="utf-8"><style>'
            'body{margin:0;background:white}svg{width:100%;height:auto;display:block;min-width:1100px}'
            '</style></head><body>' + svg + '</body></html>')
