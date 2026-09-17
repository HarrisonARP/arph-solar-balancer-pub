"""Build the demo bundle shipped with the repo, from cached weather only.

The dashboard's "Load demo case" button reads what this writes, so a reviewer can
explore a full set of results — map, dispatch, limiting subsystem, costs, PFD — without
an API key and without waiting for a simulation. Regenerate whenever the model changes
enough that the shipped numbers would misrepresent it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (RUN_JOBS, RUN_JOBS_LOCK, SHOWCASE_RESULTS,  # noqa: E402
                 _execute_showcase_job, _execute_single_site_job, _job_snapshot,
                 cached_site_entries, compact_for_demo)
from result_files import (JOB_FIELDS, collect_weather_cache,  # noqa: E402
                          export_results, import_results)

# Demo bundles are plotted, never re-analysed, so they store display columns only and
# round what they store. See compact_for_demo.
DEMO_DECIMALS = 6

DEMO_DIR = Path("data") / "demo"
# Two showcases differing in one variable only, so the pair answers "what does running
# several smaller trains in parallel buy you?" rather than confounding it with sizing.
DEMO_VARIANTS = {"parallel": 4, "single": 1}
# The demo deliberately pins settings rather than reading the app's control defaults:
# a shipped result has to stay reproducible even after a default is retuned. Long
# storage is paired H2 + CO2 because that is the cheapest of the three methods, so the
# demo shows the design at its best rather than at its strawman.
DEMO_SETTINGS = {
    "farm_mw": 10.0,
    "weather_source": "cached",
    "ninja_api_key": None,
    "latest_year": 2025,
    "reactor_count": DEMO_VARIANTS["parallel"],
    "reactor_scheduling_mode": "seasonal",
    "information_mode": "comparison",
    "short_name": "limping",
    "long_method": "h2_co2",
    "capacity_setting_mode": "manual",
    # f_OCP derates: the process is sized as nominal / (1 + f_OCP), so 0.70 builds a
    # smaller process than 0.50 did and leaves more of the farm's output as margin.
    # That is what closes the cyclic energy balance at London, which 0.50 did not.
    # f_SOCP 1.0 then installs exactly the long-storage energy the cyclic sizing asked
    # for, so the shipped demo is not one of the under-built cases the deficit warning
    # is there to catch.
    "f_ocp": 0.70,
    "f_socp": 1.0,
    "air_hx_effectiveness": 0.97,
    "initial_soc_fraction": 0.50,
    "factor_minimum": 0.0,
    "factor_maximum": 1.0,
    "factor_increment": 0.25,
    "investigate": [],
    "run_scope": "specific",
    "faults_enabled": True,
    "fault_seed": 0,
    # A visibly faulty plant. One event per subsystem per quarter, three days long,
    # running at 60% of rated capacity while it lasts — roughly four times the lost
    # capacity-hours of the gentler setting this replaces. The demo exists partly to
    # show what equipment failure costs, which a plant that rarely breaks cannot do.
    "fault_config": {name: {"months": 3, "duration_h": 72, "capacity_fraction": 0.6}
                     for name in ("battery", "hydrogen", "sabatier", "dac")},
    "assumptions": {},
}


def demo_sites() -> list[dict]:
    """Cached sites holding a long enough contiguous span to run offline."""
    return [{"site": entry["site"], "lat": entry["lat"], "lon": entry["lon"]}
            for entry in sorted(cached_site_entries(), key=lambda e: -e["lat"])
            if entry["runnable"]]


def _weather_for(sources, include_weather: bool):
    """Bundle the cached years a result depends on, unless told to leave them out."""
    if not include_weather:
        return [], []
    return collect_weather_cache(sources)


def build_single_site(site: dict, *, include_weather: bool, training_years: int,
                      evaluation_years: int) -> bytes:
    job_id = uuid4().hex
    with RUN_JOBS_LOCK:
        RUN_JOBS[job_id] = {"status": "queued", "messages": [],
                            "delivered_selection": None, "delivered_failure": False,
                            "delivered_comparison": None}
    # include_baseline defaults on, so the demo carries the do-nothing comparison too.
    _execute_single_site_job(job_id, {**DEMO_SETTINGS, "lat": site["lat"],
                                      "lon": site["lon"],
                                      "assumptions": {
                                          "weather.training_years": training_years,
                                          "weather.evaluation_years": evaluation_years}},
                             progress_callback=lambda message: print(f"    {message}",
                                                                     flush=True))
    job = _job_snapshot(job_id)
    if job["status"] != "complete":
        raise SystemExit(f"Single-site demo failed: {job.get('error')}")
    payload = {key: job[key] for key in JOB_FIELDS if key in job}
    weather, missing = _weather_for([job.get("source_metadata", {})], include_weather)
    if missing:
        raise SystemExit(f"Missing cached weather for the demo site: {missing}")
    return export_results("single_site", compact_for_demo(payload), weather,
                          decimals=DEMO_DECIMALS)


def convert_single_site(saved: Path, *, include_weather: bool) -> bytes:
    """Rebuild the single-site demo from a result file saved out of the dashboard.

    A long evaluation takes a while to run, so a case explored in the app can become
    the shipped demo without running it a second time. Weather travels with the saved
    file, so the cache on this machine is never consulted.
    """
    kind, payload, weather = import_results(saved.read_bytes())
    if kind != "single_site":
        raise SystemExit(f"{saved} holds {kind} results; the single-site demo needs a "
                         "file saved from the Single site tab.")
    return export_results("single_site", compact_for_demo(payload),
                          weather if include_weather else [], decimals=DEMO_DECIMALS)


def build_showcase(sites: list[dict], *, include_weather: bool,
                   reactor_count: int) -> bytes:
    settings = {**DEMO_SETTINGS, "reactor_count": reactor_count}
    # Rows accumulate in a module-level store keyed by site name, so a second variant
    # would otherwise inherit the first variant's rows for any site it skipped.
    with RUN_JOBS_LOCK:
        SHOWCASE_RESULTS.clear()
    for number, site in enumerate(sites, start=1):
        started = time.time()
        print(f"  [{number}/{len(sites)}] {site['site']}", flush=True)
        job_id = uuid4().hex
        with RUN_JOBS_LOCK:
            RUN_JOBS[job_id] = {"status": "queued", "messages": []}
        # One site per job so a long batch reports progress and a single bad site is
        # obvious immediately rather than at the end.
        _execute_showcase_job(job_id, [site], settings)
        errors = _job_snapshot(job_id).get("errors") or []
        with RUN_JOBS_LOCK:
            RUN_JOBS.pop(job_id, None)
        if errors:
            raise SystemExit(f"Showcase demo failed: {errors}")
        print(f"      done in {time.time() - started:.0f}s", flush=True)
    with RUN_JOBS_LOCK:
        payload = list(SHOWCASE_RESULTS.values())
    sources = [{**(row.get("source_metadata") or {}), "site": row.get("site")}
               for row in payload]
    weather, missing = _weather_for(sources, include_weather)
    if missing:
        raise SystemExit(f"Missing cached weather for: {missing}")
    return export_results("showcase", payload, weather)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="London",
                        help="Cached site to use for the single-site demo case.")
    parser.add_argument("--no-weather", action="store_true",
                        help="Leave the cached weather years out of the bundle, for "
                             "a repo that ships data/weather_cache itself.")
    parser.add_argument("--training-years", type=int, default=5,
                        help="Climatology period for the single-site demo case.")
    parser.add_argument("--evaluation-years", type=int, default=10,
                        help="Evaluation period for the single-site demo case.")
    parser.add_argument("--from-save", type=Path,
                        help="Build the single-site demo from a result file already "
                             "saved out of the dashboard, instead of running a case. "
                             "Use this to promote a run you explored in the app.")
    parser.add_argument("--variant", choices=sorted(DEMO_VARIANTS),
                        help="Build only this showcase variant; default builds both.")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-showcase", action="store_true")
    args = parser.parse_args()

    include_weather = not args.no_weather
    # Converting a saved file touches no weather cache, so a machine that has one
    # result file and nothing else can still rebuild the single-site demo.
    needs_cache = not args.skip_showcase or not (args.skip_single or args.from_save)
    sites = demo_sites() if needs_cache else []
    if needs_cache and not sites:
        raise SystemExit("No cached site holds a long enough span. Populate "
                         "data/weather_cache first.")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    if sites:
        print(f"{len(sites)} runnable cached sites: "
              + ", ".join(site["site"] for site in sites))

    if not args.skip_showcase:
        for variant, reactor_count in DEMO_VARIANTS.items():
            if args.variant not in (None, variant):
                continue
            print(f"Showcase demo ({variant}, {reactor_count} train"
                  f"{'' if reactor_count == 1 else 's'}):")
            contents = build_showcase(sites, include_weather=include_weather,
                                      reactor_count=reactor_count)
            target = DEMO_DIR / f"showcase-{variant}.json.gz"
            target.write_bytes(contents)
            print(f"  wrote {target} ({len(contents) / 1e6:.1f} MB)")

    if not args.skip_single:
        if args.from_save:
            if not args.from_save.is_file():
                raise SystemExit(f"No result file at {args.from_save}.")
            print(f"Single-site demo: converting {args.from_save}")
            contents = convert_single_site(args.from_save,
                                           include_weather=include_weather)
        else:
            chosen = next((site for site in sites
                           if site["site"].casefold() == args.site.casefold()), None)
            if chosen is None:
                raise SystemExit(f"No cached site called {args.site!r}. "
                                 f"Available: {[site['site'] for site in sites]}")
            print(f"Single-site demo: {chosen['site']}")
            contents = build_single_site(chosen, include_weather=include_weather,
                                         training_years=args.training_years,
                                         evaluation_years=args.evaluation_years)
        (DEMO_DIR / "single_site.json.gz").write_bytes(contents)
        print(f"  wrote {DEMO_DIR / 'single_site.json.gz'} ({len(contents) / 1e6:.1f} MB)")
