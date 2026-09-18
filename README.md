# Solar Balancer — user guide

Solar Balancer is a toy digital twin of an off-grid solar-to-methane plant. It sizes the
plant and its storage for a location, dispatches it hour by hour against real or
synthetic weather, and reports what it produced and what that cost. Built for the
SoTA/Rivan *Intermittent Abundance* competition.


Disclaimer

This is a toy model used to demonstrate the parameters influencing power-to-X (P2X) chemical production, intended for research use only. Calculated values and input parameters are indicative only and should not be used for real plant design or construction, operational decisions, or investment decisions. The model is provided without any guarantee of accuracy, completeness, or fitness for a particular purpose. The author bears no responsibility for any loss, damage, or consequences arising from its use or reliance on its outputs. No responsibility is taken for any damage to computer equipment resulting from executing the code in this repository.


> **Numbers are illustrative.** `costs.json` is marked `illustrative_dummy` and the
> physical defaults are indicative.

---

## Contents

- [Five-minute start](#five-minute-start) — results without running anything
- [Installation](#installation)
- [Running a single-site case](#running-a-single-site-case)
- [Running a showcase](#running-a-showcase)
- [Three things to know before changing settings](#three-things-to-know-before-changing-settings)
- [Control reference](#control-reference)
- [What happens during a faulted imperfect run](#what-happens-during-a-faulted-imperfect-run)
- [Known model limitations](#known-model-limitations)
- [Addenda](#addenda) — licence, AI use and citations

---

## Five-minute start

No API key, no simulation.

1. Install (below), run `python app.py`, open <http://127.0.0.1:8050/>.
2. Click the amber **Load demo case** under the title.

That restores ten European sites on real reanalysis weather plus one fully worked site
with pre-calculated hourly energy dispatching.

For the worked site's plots, open **Single site**, pick the entry in **Completed case**,
and press **Load selected dispatch plot**.

**Loading the demo is also what puts the weather on disk.** The bundles carry the
reanalysis years with them, and loading writes all ten sites into `data/weather_cache/`.
Until you have done it once, **Use cached weather** finds nothing and every run needs an
API key. Afterwards you can run your own cases on any of the ten sites with no key and
no downloads — see below.

Annotated example calculations showing the derivation of the plant energy requirements, and energy dispatch models, are included in the Jupyter notebook `sab_notebook.ipynb`. Its first six sections are algebra on a one-kilogram basis and run in seconds; from section 7 it dispatches a weather record hour by hour and takes one to two minutes on a current machine, five to ten on an older one. The notebook deliberately uses one training year and five evaluation years rather than the five and ten behind every shipped result, purely so it runs quickly - the procedure it demonstrates is the same.

---

## Installation

**Python 3.11 or newer.**

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8050/>; Ctrl+C to stop. Dependencies are CoolProp, Dash, NumPy,
pandas, Plotly and requests. To check the install — no extra packages, no network:

```powershell
python -m unittest discover -s tests
```

The output should read: 125 tests, 2 skipped. 
Expect **under ten minutes** on a current machine and **half an hour or so** on an older one — a 2017 dual-core machine took 36 minutes. The two skips are the notebook tests, which need `nbclient` and `nbformat` (not included in requirements.txt).

### Rebuilding the demo bundles

`scripts/build_demo.py` regenerates the three files in `data/demo/` from cached weather,
so every shipped number is reproducible; it takes about half an hour on a current
machine, two to three on an older one, and needs no API key.

### Optional: a Renewables.ninja API key

Needed only to download **new** weather for an uncached location. Demo bundles, cached
years and synthetic weather all work without one.

A key can be requested by making a free account at
<https://www.renewables.ninja/register>. It is never written into results or into any
tracked file.

Put it in `.env` at the project root:

```text
RENEWABLES_NINJA_TOKEN=your_token_here
```

You can paste it into the app's **Renewables.ninja API key** box
instead; `.env` takes priority.

---

## Running a single-site case

Runs the model for real on cached weather. This needs no Renewables.ninja API key and makes no API calls.

1. Press **Load demo case** first if you have not already — that is what puts the
   cached weather on disk. See the five-minute start above.
2. **Single site** tab → **Weather source** → **Cached only**.
3. Press **Use cached weather**. Coordinates jump to a cached site; the status line lists
   what is available. London is a good default.
4. Leave the defaults: 10 MW, 4 parallel trains, seasonal scheduling, *Perfect vs
   imperfect*, limping, battery long storage.
5. Press **Run selected case**.

From the demo data sets you can adjust the site parameters and operating strategies, or
select one of the other pre-loaded sites, and run that instead. Ticking **Include in
test cases** under any group of settings runs every option in that group, one at a time,
which multiplies the runtime accordingly.

A ten-year evaluation takes a minute or two on a current machine and eight to ten
minutes on an older one: the hourly dispatch is a sequential loop that uses **one
core**, so single-core speed sets the pace and extra cores do not help. Then choose
the case in **Completed case** and press **Load selected dispatch plot**. For faults,
tick **Enable stochastic faults** before running and set **Displayed result category**
to *Imperfect with faults* after.

For testing/debugging without use of cached data or an API key, a synthetic weather profile can also be generated for a selected site by selecting 'Offline synthetic'. These are purely for development purposes and any calculated system behaviour using synthetic weather data should not be treated as valid outputs.

> Faults are measured against a forecast, so ticking them while in *Perfect only* switches
> **Information mode** to *Perfect vs imperfect* and says so. Choosing *Perfect only*
> again is respected, with a note that faults will not apply.

---

## Running a showcase

One case per location, colouring a map of Europe.

**Pre-computed:** open **European showcase** and press **Load parallel reactor showcase**
or **Load single reactor showcase**. Same ten sites, same sizing, same cached weather —
only the train count differs, four or one. They share site names, so loading one replaces
the other.

Recolour the dots with the metric radio buttons, tick **Show advanced metrics** for the
fuller list, click any dot for its detail panel.

The faint grid of coordinate targets is how you pick an unmodelled site; it also sits
close enough to catch the pointer when you only want to read results. Tick **Hide the
coordinate grid** to remove it — result dots stay clickable. Remembered in your browser.

**Your own:** **Use cached weather** (selects every cached site with enough contiguous
years) → optionally adjust **Run settings (mirrors the main tab)** → **Run N cases**.

Roughly a minute and a half per location over ten years on a current machine, or
eight on an older one, much the same whether it runs four trains or one. Results
survive a page refresh but not an app restart — use **Save showcase results**.

> Showcase runs always compute all three information and fault variants regardless of
> **Information mode**, so no map metric is ever blank.

---

## Three things to know before changing settings

### 1. Renewables.ninja downloads need an API key

Selecting it without a key shows a red warning and the run fails. Use **Offline
synthetic** or **Cached only** instead. The limit is 50 requests per rolling hour and one
request is one year for one location, so a fifteen-year site costs fifteen calls.
Estimated remaining quota is shown next to the weather source, but does not persist between sessions.

### 2. Some features are experimental

| Feature | Where | Status |
| --- | --- | --- |
| **Perfect daily switching** | Reactor scheduling | Commits trains a day at a time with perfect knowledge of that day, which can behave unpredictably once combined with imperfect weather information. |
| **Methane product storage** | Results: long storage | Buffers the product instead of banking energy upstream; the vessel is sized in advance and can fill up and curtail. |

Auto-calibration of capacities was another and has been withdrawn from the dashboard: it
minimised *perfect-information* LCOM on a coarse grid, tuning f_OCP and f_SOCP to a
weather year the operator could not have known. The code remains in the repository and
under test, but is not currently reachable from the dashboard.

Every shipped bundle uses seasonal scheduling, manual capacities and H2 + CO2 storage, so
none of the above is behind any published number.

### 3. Costs are placeholders

Every economic output comes from `costs.json`, marked `illustrative_dummy`. Cost *ratios*
between configurations mean more than the absolute figures; levelised costs are for
ranking, not budgeting. Costs and assumed equipment efficiencies can be updated with reference values from literature or empirical data on the 'System & economic parameters' tab.

---

## Control reference

List of controls that change what the model computes. Display selectors (**Displayed result
category**, map metric buttons, comparison filters) and save/load buttons change what you
see, not what is calculated.

### Single site — site & weather

| Control | What it does |
| --- | --- |
| **Latitude / Longitude** | Where the plant is; also set by clicking the map. Determines solar resource and everything downstream. |
| **Solar farm (MW)** | Installed PV. The whole plant is sized from this — the model asks what process this farm can sustain, not what farm a process would need. A 10 MW farm is used as a default basis value|
| **Renewables.ninja API key** | Read only when that source is selected. Keys pasted into `.env` are prioritised over the entry in the box. |
| **Latest year (API)** | Most recent year to fetch or replay; the evaluation period ends here and training sits immediately before. **Ignored** for synthetic weather, which uses its own `synthetic.*` assumptions. |
| **Use cached weather** | Jumps to a cached site and reports what is available. |

**Weather source** — *Renewables.ninja* is real MERRA-2 reanalysis, one year per request,
cached, **key required**. *Cached only* replays years on disk and never contacts the API.
*Offline synthetic* is deterministic solar geometry, no network or key, **for quick
testing not siting claims**: it carries no regional cloud climatology, so it under-states
how much sunnier southern Europe is than the maritime north-west, and it runs its own
fixed period (six years from 2010, evaluating the last) rather than the fifteen ending at
**Latest year**. Synthetic results are not comparable with Ninja ones, but can be useful for model testing and debugging.

> **Long evaluation periods make large files.** Every hour is kept so the dispatch plots
> can show it: roughly 4.7 MB per evaluation year per result category, and a comparison
> run carries three or four, so 14–19 MB per evaluation year. A save has a 256 MB
> ceiling, reached at about 14 evaluation years across four categories, 18 across three.
> The projected size appears beside **Save single-site results** before you click. Long
> periods still *run*; the limit is on saving.

### Single site — base case

| Control | What it does |
| --- | --- |
| **Parallel trains** | How many equal Sabatier trains the plant splits into. More trains give finer turndown. *Alpha.* |
| **Initial storage SOC fraction** | How full storage is at hour zero, as a fraction of installed energy. Applied to batteries and to both inventories in paired gas storage.|
| **Include in test cases** | Tick boxes under operating strategy, long storage, parallel trains and reactor scheduling. Adds that parameter to an **Investigate** run. Selecting a large number of parameters to test in a given run will take a long time and produce very large output files.|

**Reactor scheduling** — The operating strategy used to decide when to switch reactors off and on. *Seasonal parallelisation*, fixed per-season reactor train targets (based approximately on the ratio of solar output between seasons), the
default and what every shipped result uses; or *Perfect daily switching* (experimental, attempts to schedule number of reactors on a daily basis based on the deployment heuristics described in Fulham et al. 2024 doi.org/10.1039/D4EE00933A).

**Information mode** — *Perfect only* is one dispatch with perfect foresight, fast, and
**disables faults entirely**. *Perfect vs imperfect* adds a climatology-forecast dispatch
and a third with faults if enabled, which is what makes the forecast- and fault-penalty
metrics available.

**Results: operating strategy** — overnight and in poor weather: *Through night* (stay
hot and producing methane through the night), *Limping* (minimum sustaining load to keep the calcium looping reactors and Sabatier reactor at temperature; the default), *Hard shutdown* (shut down all hot process equipment every night, allow to cool to ambient temperature,
then spend additional energy to reheat on restart).

**Results: long storage** — the carrier or storage mode used to achieve seasonal energy balancing. *Battery*, *Hydrogen* and *H2 + CO2 gas storage* bank **energy** upstream of
the reactors, to be spent later running the plant. For H2 + CO2 storage, excess power is used to produce additional H2 and CO2 in a 4:1 molar ratio as required for the Sabatier reaction, then used as a feed to the Sabatier reactor when insufficient power is available. Any additional power available once CO2 storage is full is used to generate supplementary H2, for use in a small fuel cell to generate auxiliary power.

*Methane product storage*
(experimental) differs in kind: nothing is banked upstream and the methane itself is
buffered in a vessel sized in advance from the training record, then released at a
constant rate. It is the only method that buffers the gate — under the other three,
methane leaves as it is made and output floats with the weather — and because the vessel
is sized in advance it can become full, curtailing any excess power that cannot be stored as methane. A legacy feature not included in the current build allowed for short term methane buffering as an extra system, in order to achieve constant output over the course of each day for limping and hard shutdown strategies, but resulted in buggy interactions with long-term energy balancing.

### Single site — capacity factors

| Control | What it does |
| --- | --- |
| **f_OCP** | Derating overcapacity factor: nominal plant size is divided by `1 + f_OCP`, so a **larger** f_OCP means a **smaller** process for the same farm, held in reserve against bad days. |
| **f_SOCP** | Installed-to-required ratio for long-term storage energy as estimated from sizing calculations. 1.00 installs exactly what the cyclic sizing asked for; 0.75 under-sizes by a quarter; 1.25 oversizes by a quarter |
| **Factor range min / max / increment** | Display only — rescales the sliders, changes nothing calculated. |
| **Air/exhaust exchanger effectiveness** | Fraction of the carbonator air-preheat duty recovered from depleted exhaust. Dry calcium looping captures CO2 on hot solids, so the entire air stream - about 2,200 kg per kg of CO2 at 400 ppm - must reach the carbonator temperature. **The most influential single assumption in the model:** E_req scales with whatever this does *not* recover. At the default 0.97 the residual air duty is still about 39% of E_req. |

### Single site — faults

**Enable stochastic faults** turns on the third dispatch variant (switching **Information
mode** if needed). **Random seed** reproduces a schedule exactly, which is what makes
fault comparisons between configurations fair.

**Per-subsystem sliders**, three each for battery, hydrogen, Sabatier and DAC: *mean
interval* in months (**0 disables**), *mean duration* in hours, and *mean retained
capacity* during an event — 0.8 leaves 80% working, 0 is total loss.

### Single site — run buttons

**Run selected case** runs one case at current settings. **Investigate N selected
parameters** runs the base case plus one case per value of each ticked parameter, so the
runtime is that of a single run multiplied by the number of cases.

### European showcase

| Control | What it does |
| --- | --- |
| **Map dots and faint points** | Click to select a location, click again to deselect. |
| **Use cached weather** | Selects every cached site with enough contiguous years. 10 sites are pre-loaded in the code bundle.|
| **Run settings (mirrors the main tab)** | f_OCP, f_SOCP, operating strategy, long storage, weather source, fault seed and all twelve fault sliders. Changing either copy updates the other. |
| **Run N cases** / **Clear selection** | Run one case per selected location; clear without touching results already on the map. |

### System & economic parameters

**Assumptions table** — every physical, thermal, storage and economic constant, editable,
with units and descriptions. Values owned by the Single site tab (farm size, f_OCP,
f_SOCP, train count, initial SOC, exchanger effectiveness) are read-only here so two
controls can never disagree. **Save parameters** applies edits to subsequent runs; they
do nothing until pressed.

### Process flow diagram, Sizing, Citations

No controls that change results. The PFD tab's **Completed case** and **Result category**
selectors choose which run annotates the diagram.

---

## What happens during a faulted imperfect run

*Perfect vs imperfect* with faults enabled runs **four** dispatches over the same
weather. The app reports each step as it happens.

**1. Weather.** Years are loaded and split into a training period (5 years by default) and an out-of-sample
evaluation period (the remainder of the selected period, 10 years by default). Only the evaluation period is dispatched.

**2. Forecast.** An hourly climatology from the *training* years alone by averaging the profiles of each calendar day. The plant knows
what a typical 14 March looks like, not the actual one it will meet.

**3. Fault schedule.** Drawn independently per subsystem before any dispatch, from the
seed and that subsystem's sliders. Each hour is an independent trial with probability
`1 − exp(−1 / (hours_per_month × mean_interval))`, so events arrive as a Poisson process.
Duration and retained capacity are drawn from normal distributions about the slider
values with 25% standard deviation, duration at least one hour. Overlapping events on one
subsystem take the **worst** capacity. Fixed from here, so every dispatch below meets
identical failures.

**4. Plant sizing.** Stoichiometry and energy requirement per kg of methane, then the
nominal plant size the farm sustains, derated by `1 + f_OCP`.

**5. Storage sizing — twice, for two different plants.** Storage is sized from the range
of a state-of-charge trajectory: the cyclic requirement that lets the year close.
Long-term storage is then scaled by f_SOCP. Transfer *power* is not constrained here; it
is measured after dispatch and used for costing (e.g. for sizing electrolysers).

This happens twice. The perfect-information case is sized on the evaluation period it
actually meets; the forecast-driven cases — imperfect, with faults, and the no-storage
baseline — on the *training* record, because a real designer specifies the plant before
the decade it must survive. Capacities, throughput and equipment ratings all follow.
The gap between the perfect case and the rest therefore carries the cost of sizing blind
as well as dispatching blind: **they are not the same plant.**

**6. Commitment schedule.** Built from the *forecast*, not actual weather — two plants,
so two schedules. Within a plant every case shares one, so cases commit the same
equipment on the same days and differ only in what the weather and equipment then do. The attempted energy transfer in and out of storage is committed at this stage.

**7. Perfect information.** Meets the actual weather knowing it exactly, with no equipment faults. The
ideal the other cases are measured against.

**8. Imperfect forecast.** Plans against climatology, then meets actual weather. The gap
from step 7 is the cost of not knowing the future in *both* design and operation, as the plant design will be different from the perfect information case (from preliminary testing, sites could end up oversized or undersized when designed using imperfect forecasting, resulting in either curtailment or worse than expected output). Imperfect weather forecasting generally results in a less productive and more expensive plant than perfect forecasting, but this is not a deterministic constraint, and in some cases unusual weather patterns in the training or evaluation period can result in a slightly more efficient plant (albeit well within the margin of uncertainty for any given case).

**9. Imperfect with faults.** As step 8 plus the schedule from step 3. Each faulted hour
derates that subsystem against the capacity the perfect case established, so 0.8 means
genuinely 80% of what that equipment could have done. The gap from step 8 is the cost of
equipment failure. The faulted case should always be less productive and more expensive than the case without faults.

**10. No-storage baseline.** The same plant with **no storage and no scheduling**: flat
out whenever the sun allows, curtail what cannot be absorbed, shut down at night. It
carries the same faults as step 9, because breakdowns do not care how a plant is
dispatched. The **No-storage baseline production ratio** is this divided by step 9.
This baseline ratio should always be at or below 1.0 — adding energy balancing should increase methane output — though it might not always reduce cost, if storage is sufficiently expensive. The *forecast + fault* production ratio is a separate metric and can exceed 1.0, because the forecast-driven cases run a differently sized plant.

E.g.
Four-train showcase: 0.86 at London, 0.67 at Tromsø — the largest gain of the ten sites.
Single-train: 0.62 and 0.49, further apart, as one train cannot follow the
weather the way four committed in blocks can, so scheduling has more to recover.

**11. Economics.** CAPEX, OPEX and levelised cost of methane for each of the four cases, using the
storage transfer powers observed during dispatch.

Four comparable annual figures from one run, separating imperfect information (7→8),
equipment failure (8→9), and the value of storage and scheduling (10→9). Only 8→9 and
10→9 compare like plants with like; levelised cost of methane is generally a more useful metric than raw CAPEX or OPEX.

---

## Known model limitations

**The forecast never refreshes.** The climatology is averaged once from the training
years and applied unchanged across the whole evaluation period, so a plant evaluated over
2016–2025 is still forecasting from 2011–2015 in its final year — a record ten to
fourteen years stale. A real operator would fold each year's outturn back in as it
arrived.

This matters most when comparing sites. The forecast cost ratio barely tracks forecast
noise at all; it is almost entirely a statement about plant size, correlating +1.00 with
the ratio of installed capital between the two cases. It measures a decadal sampling
accident rather than how predictable a site is, so read cross-site rankings built on it
with care, and use the production ratio to judge forecast quality.

An improvement would be rolling climatology: forecast each evaluation year from the years
immediately preceding it. That affects only the dispatch forecast. It must **not** be
applied to sizing, which is deliberately fixed at the build year.

A more significant improvement, and a topic of further study, would be using a longer or synthetically extended
design record, and a sizing rule stated against a return period — size for the worst
winter in N years — rather than whatever the record happened to contain. At present this is partially accounted for by the overcapacity factors f_OCP and f_SOCP, but it is up to the user to guess how much extra capacity is needed.

## Addenda

This model was created by ARP Harrison, and is distributed free of charge under the Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0). See [`LICENSE.md`](LICENSE.md) in the repository root for the terms and for the third-party components this work depends on. Generative artificial intelligence tools (namely OpenAI Codex and Claude Code agents, predominantly running the GPT-5.6 and Opus 5 model families, respectively) were used during code development and documentation.

This work makes use of the Renewables.ninja retroanalysis tool, and the CoolProp thermodynamic libraries - relevant citations are given below, and should be included in any derivative works using code from this repository.

CoolProp
    Bell, I. H., Wronski, J., Quoilin, S. and Lemort, V. (2014). Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp. Industrial & Engineering Chemistry Research, 53(6), 2498-2508.
https://doi.org/10.1021/ie4033999

Renewables.ninja (solar PV)
    Pfenninger, S. and Staffell, I. (2016). Long-term patterns of European PV output using 30 years of validated hourly reanalysis and satellite data. Energy, 114, 1251-1265.
https://doi.org/10.1016/j.energy.2016.08.060

Renewables.ninja (companion paper)
    Staffell, I. and Pfenninger, S. (2016). Using bias-corrected reanalysis to simulate current and future wind power output. Energy, 114, 1224-1239.
https://doi.org/10.1016/j.energy.2016.08.068

MERRA-2 reanalysis
    Gelaro, R. et al. (2017). The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). Journal of Climate, 30(14), 5419-5454.
https://doi.org/10.1175/JCLI-D-16-0758.1


Many thanks are extended to Dr G Fulham for helpful discussions regarding modelling approaches for power-to-X chemical production.
