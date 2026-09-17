"""Versioned, data-only JSON exports for dashboard results."""

from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
import gzip
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
import pandas as pd

from model import (DEFAULT_CACHE_DIR, CaseResult, PlantEnergyResult, SimulationResult,
                   StorageSizingResult, validate_hourly_profile)

MAX_FILE_BYTES = 256 * 1024 * 1024
# 1: every frame as a to_json(orient="table") string. 2: columnar frames.
FORMAT_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
# Measured on version-2 columnar output at full precision: one stored float costs
# about thirteen bytes of JSON. Used only to warn before a save, where being roughly
# right early beats being exactly right too late.
BYTES_PER_STORED_VALUE = 13
TYPES = {cls.__name__: cls for cls in (
    CaseResult, PlantEnergyResult, SimulationResult, StorageSizingResult)}
JOB_FIELDS = ("results", "case_labels", "source_metadata", "evaluation_period",
              "information_mode", "assumptions", "faults_enabled", "fault_scenario",
              "optimized_factors", "run_scope", "inputs")


def normalize_showcase_ratio(row):
    """Use the complete realized LCOM penalty against the fault-free perfect case."""
    row = dict(row)
    outputs = row.get("outputs") or {}
    perfect = outputs.get("economics_perfect") or {}
    faulted = outputs.get("economics_imperfect_with_faults")
    imperfect = outputs.get("economics_imperfect")
    realized = faulted if faulted is not None else imperfect
    denominator = perfect.get("lcom_usd_per_kg_ch4")
    numerator = (realized or {}).get("lcom_usd_per_kg_ch4")
    row["forecast_cost_ratio"] = (
        numerator / denominator
        if isinstance(numerator, (float, int)) and math.isfinite(numerator)
        and isinstance(denominator, (float, int)) and math.isfinite(denominator)
        and denominator > 0 else None
    )
    row["forecast_cost_ratio_basis"] = (
        "Imperfect weather + faults / perfect information, no faults"
        if faulted is not None else
        "Imperfect weather / perfect information (faults disabled)"
        if imperfect is not None else
        "Unavailable: perfect-only run or legacy file without category economics"
    )
    return row


def safe_inputs(values):
    """Exclude credentials recursively from exported input/metadata dictionaries."""
    if isinstance(values, dict):
        return {key: safe_inputs(value) for key, value in values.items()
                if not any(secret in key.lower() for secret in
                           ("token", "api_key", "api-key", "password", "authorization"))}
    if isinstance(values, (list, tuple)):
        return [safe_inputs(value) for value in values]
    return values


def _encode_index(index):
    """Describe a frame's index without spelling out every label.

    An hourly result carries one evenly spaced UTC timestamp per row, so the whole
    index is three numbers rather than 87,672 ISO strings. Anything irregular falls
    back to listing epoch nanoseconds, which is still a third of an ISO timestamp.
    """
    if isinstance(index, pd.DatetimeIndex):
        stamps = index.asi8
        step = int(stamps[1] - stamps[0]) if len(stamps) > 1 else 0
        regular = len(stamps) > 1 and bool((np.diff(stamps) == step).all()) and step > 0
        described = ({"start": int(stamps[0]), "step_ns": step, "count": len(stamps)}
                     if regular else
                     {"epoch_ns": [int(stamp) for stamp in stamps]})
        return {"kind": "datetime", "name": index.name,
                "tz": str(index.tz) if index.tz is not None else None, **described}
    if isinstance(index, pd.RangeIndex):
        # The default index of a frame carrying no rows, such as the daily table a
        # compacted simulation drops. Three numbers, and it comes back as itself.
        return {"kind": "range", "name": index.name, "start": index.start,
                "stop": index.stop, "step": index.step}
    return {"kind": "plain", "name": index.name,
            "values": [_encode(label) for label in index.tolist()]}


def _encode_column(series, decimals):
    """Encode one column as a bare list, keeping non-finite floats distinguishable.

    ``allow_nan=False`` rules out bare NaN/Infinity tokens in the document, and a
    null alone cannot tell a missing hour from an infinite one, so the two
    infinities travel as strings.
    """
    if series.dtype.kind == "f":
        values = series.to_numpy(dtype=float, copy=False)
        if decimals is not None:
            values = np.round(values, decimals)
        finite = np.isfinite(values)
        if finite.all():
            return values.tolist()
        return [value if is_finite else
                (None if math.isnan(value) else
                 ("Infinity" if value > 0 else "-Infinity"))
                for value, is_finite in zip(values.tolist(), finite.tolist())]
    if series.dtype.kind in "iub":
        return series.to_numpy(copy=False).tolist()
    return [_encode(value, decimals) for value in series.tolist()]


def _encode_frame(frame, decimals):
    """Store a frame column-wise, so a column name costs one copy and not one per row.

    ``to_json(orient="table")`` writes an object per row, repeating all forty column
    names — averaging 23 characters each — for every hour, and the result is then
    escaped a second time as a JSON string. Column-wise, a simulation-year costs a
    few megabytes rather than fifteen.
    """
    return {"_type": "Frame", "index": _encode_index(frame.index),
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(frame[column].dtype) for column in frame.columns],
            "data": [_encode_column(frame[column], decimals) for column in frame.columns]}


def _encode(value, decimals=None):
    if isinstance(value, pd.DataFrame):
        return _encode_frame(value, decimals)
    if is_dataclass(value):
        return {"_type": type(value).__name__, "fields": {
            item.name: _encode(getattr(value, item.name), decimals)
            for item in fields(value) if item.name != "output_dir"}}
    if isinstance(value, dict):
        return {key: _encode(item, decimals) for key, item in safe_inputs(value).items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, decimals) for item in value]
    if isinstance(value, np.generic):
        return _encode(value.item(), decimals)
    if isinstance(value, float) and not math.isfinite(value):
        return {"_type": "float", "value": str(value)}
    if isinstance(value, (Path, datetime)):
        return str(value)
    return value


_NON_FINITE = {"Infinity": math.inf, "-Infinity": -math.inf}


def _decode_index(described):
    if not isinstance(described, dict):
        raise ValueError("Invalid result table index.")
    if described.get("kind") == "datetime":
        if "epoch_ns" in described:
            stamps = np.asarray(described["epoch_ns"], dtype="int64")
        else:
            start, step = int(described["start"]), int(described["step_ns"])
            count = int(described["count"])
            if count < 0 or count > MAX_FILE_BYTES // 8:
                raise ValueError("Invalid result table length.")
            stamps = start + np.arange(count, dtype="int64") * step
        index = pd.DatetimeIndex(stamps.astype("datetime64[ns]"), name=described.get("name"))
        timezone = described.get("tz")
        return index.tz_localize("UTC").tz_convert(timezone) if timezone else index
    if described.get("kind") == "range":
        return pd.RangeIndex(start=int(described["start"]), stop=int(described["stop"]),
                             step=int(described["step"]), name=described.get("name"))
    return pd.Index([_decode(label) for label in described.get("values", [])],
                    name=described.get("name"))


def _decode_column(values, dtype):
    if not isinstance(values, list):
        raise ValueError("Invalid result table column.")
    if dtype.startswith("float"):
        return pd.Series([_NON_FINITE.get(value, value) if isinstance(value, str)
                          else value for value in values], dtype="float64")
    series = pd.Series([_decode(value) for value in values])
    try:
        return series.astype(dtype)
    except (TypeError, ValueError):
        # A dtype that no longer round-trips is not worth failing a whole load for;
        # the plots read these as numbers or labels either way.
        return series


def _decode_frame(value):
    columns, dtypes = value["columns"], value["dtypes"]
    if not isinstance(columns, list) or len(dtypes) != len(columns):
        raise ValueError("Invalid result table schema.")
    index = _decode_index(value["index"])
    if not columns:
        # A table with no columns at all, such as the daily frame a compacted
        # simulation drops. Naming an empty column list would hand back an
        # object-dtype axis where the original had the default one.
        return pd.DataFrame(index=index)
    frame = pd.DataFrame(
        {column: _decode_column(data, dtype).to_numpy()
         for column, data, dtype in zip(columns, value["data"], dtypes)},
        index=index, columns=columns,
    )
    # Going through to_numpy above keeps the decoded columns from being aligned
    # against the index, but it also drops the pandas dtype: with rows present the
    # constructor infers an equivalent one, and with none it cannot. pandas 3 reads
    # strings as "str" where pandas 2 read "object", so restore whatever was stored
    # rather than letting an empty table come back differently typed.
    for column, dtype in zip(columns, dtypes):
        if str(frame[column].dtype) != dtype:
            try:
                frame[column] = frame[column].astype(dtype)
            except (TypeError, ValueError):
                # Same reasoning as _decode_column: a dtype that no longer exists is
                # not worth failing a whole load for.
                pass
    if not frame.empty and (not isinstance(frame.index, pd.DatetimeIndex)
                            or not frame.index.is_unique):
        raise ValueError("Result tables must have unique datetime indices.")
    return frame


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("_type")
    if kind == "Frame":
        return _decode_frame(value)
    if kind == "DataFrame":
        # Version 1 wrote every frame as a row-per-object table. Files saved before
        # the columnar format still load.
        frame = pd.read_json(StringIO(value["table"]), orient="table")
        if not frame.empty and (not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique):
            raise ValueError("Result tables must have unique datetime indices.")
        return frame
    if kind == "float":
        if value["value"] not in {"inf", "-inf", "nan"}:
            raise ValueError("Invalid numeric marker.")
        return float(value["value"])
    if kind in TYPES:
        return TYPES[kind](**{key: _decode(item) for key, item in value["fields"].items()})
    if kind is not None:
        raise ValueError("Unknown result data type.")
    return {key: _decode(item) for key, item in value.items()}


def collect_weather_cache(sources, cache_dir=DEFAULT_CACHE_DIR):
    """Gather the cached weather years the given sources reference.

    Returns ``(entries, missing)``. A source whose cached years have since been
    deleted is reported in ``missing`` rather than aborting the whole export, so
    losing one site's weather does not block saving every other result. Each
    source may carry a ``site`` label, which is what the report names.
    """
    entries, missing = {}, {}
    for source in sources:
        key = source.get("cache_key")
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{16}", key):
            continue
        label = source.get("site") or key
        for year in source.get("years", []):
            year = int(year)
            csv = Path(cache_dir) / key / f"{year}.csv"
            metadata = csv.with_suffix(".metadata.json")
            if not csv.exists():
                missing.setdefault(label, []).append(year)
                continue
            entries[(key, year)] = {
                "cache_key": key, "year": year, "csv": csv.read_text(encoding="utf-8"),
                "metadata": safe_inputs(json.loads(metadata.read_text(encoding="utf-8")))
                    if metadata.exists() else {},
            }
    return list(entries.values()), [
        f"{label} ({len(years)} year{'s' if len(years) != 1 else ''})"
        for label, years in sorted(missing.items())
    ]


def validate_weather_cache(entries):
    if not isinstance(entries, list):
        raise ValueError("Invalid weather cache section.")
    validated = []
    for entry in entries:
        key, year = entry["cache_key"], entry["year"]
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{16}", key):
            raise ValueError("Invalid weather cache key.")
        if type(year) is not int or not 1900 <= year <= 2200:
            raise ValueError("Invalid weather cache year.")
        frame = pd.read_csv(StringIO(entry["csv"]), parse_dates=["timestamp"]).set_index("timestamp")
        frame = validate_hourly_profile(frame, expected_year=year)
        metadata = entry["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Invalid weather metadata.")
        validated.append({"cache_key": key, "year": year,
                          "csv": frame.rename_axis("timestamp").to_csv(),
                          "metadata": safe_inputs(metadata)})
    return validated


def _already_cached(entry, cache_dir) -> bool:
    """Whether this year is on disk already, checked without trusting the entry.

    A cache key is a hash of the request parameters and the year is in the filename, so
    a year already present holds the same data by construction. Anything that fails
    these checks is reported as absent so validation below rejects it properly, rather
    than being used to probe the filesystem.
    """
    if not isinstance(entry, dict):
        return False
    key, year = entry.get("cache_key"), entry.get("year")
    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{16}", key):
        return False
    if type(year) is not int or not 1900 <= year <= 2200:
        return False
    return (Path(cache_dir) / key / f"{year}.csv").exists()


def restore_weather_cache(entries, cache_dir=DEFAULT_CACHE_DIR):
    # Years already on disk are dropped before validation, not after: rewriting them
    # reformats every file, which turns each load into a large spurious diff where the
    # cache is tracked in git, and re-parsing them only to discard the result is the
    # single slowest step in loading a bundle.
    present = sum(1 for entry in entries if _already_cached(entry, cache_dir))         if isinstance(entries, list) else 0
    # Validate every remaining year before writing any files; uploaded filenames are
    # never used.
    entries = validate_weather_cache(
        [entry for entry in entries if not _already_cached(entry, cache_dir)]
        if isinstance(entries, list) else entries
    )
    for entry in entries:
        directory = Path(cache_dir) / entry["cache_key"]
        directory.mkdir(parents=True, exist_ok=True)
        for suffix, content in (("csv", entry["csv"]),
                                ("metadata.json", json.dumps(entry["metadata"]))):
            destination = directory / f"{entry['year']}.{suffix}"
            temporary = directory / f"{uuid4().hex}.tmp"
            try:
                temporary.write_text(content, encoding="utf-8")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return len(entries) + present


def _payload_frames(payload):
    """Every hourly/daily table a payload would write, whatever its kind."""
    results = payload.get("results") if isinstance(payload, dict) else None
    for result in (results or {}).values():
        for simulation in (getattr(result, "perfect", None),
                           getattr(result, "imperfect", None),
                           getattr(result, "imperfect_with_faults", None),
                           getattr(result, "baseline", None)):
            if simulation is None:
                continue
            for frame in (simulation.hourly, simulation.daily):
                if isinstance(frame, pd.DataFrame):
                    yield frame


def estimate_export_bytes(payload):
    """Approximate the uncompressed export size without building the document.

    Counts the stored values rather than guessing from the run settings, so it
    stays right when a case retains fewer columns than another.
    """
    return sum(frame.shape[0] * frame.shape[1] for frame in _payload_frames(payload)
               ) * BYTES_PER_STORED_VALUE


def describe_export_size(payload):
    """A one-line size estimate for the save button, or None when there is nothing."""
    frames = list(_payload_frames(payload))
    if not frames:
        return None
    estimate = sum(frame.shape[0] * frame.shape[1]
                   for frame in frames) * BYTES_PER_STORED_VALUE
    rows = max((frame.shape[0] for frame in frames), default=0)
    detail = (f"about {estimate / 1e6:,.0f} MB from {len(frames)} table"
              f"{'' if len(frames) == 1 else 's'} of up to {rows:,} hours")
    if estimate > MAX_FILE_BYTES:
        return (f"Saving would produce {detail}, over the "
                f"{MAX_FILE_BYTES / 1e6:,.0f} MB limit. Shorten the evaluation "
                f"period, run perfect-only, or save fewer cases together.")
    if estimate > MAX_FILE_BYTES // 2:
        return f"Saving will produce {detail} — close to the limit."
    return f"Saving will produce {detail}."


def describe_export_overflow(actual_bytes):
    return (f"Export is {actual_bytes / 1e6:,.0f} MB uncompressed, over the "
            f"{MAX_FILE_BYTES / 1e6:,.0f} MB limit. Shorten the evaluation period, "
            f"run perfect-only instead of comparison, or save fewer cases together.")


def export_results(kind, payload, weather_cache=None, *, decimals=None):
    """Serialise results to a gzipped document.

    ``decimals`` rounds stored floats, which is for bundles built purely to be
    plotted; leave it None to keep every saved value exactly as it was computed.
    """
    if kind == "showcase":
        payload = [normalize_showcase_ratio(row) for row in payload]
    document = {"format": "solar-balancer-results", "version": FORMAT_VERSION,
                "kind": kind, "created_at": datetime.now(UTC).isoformat(),
                "payload": _encode(payload, decimals),
                "weather_cache": _encode(weather_cache or [], decimals)}
    raw = json.dumps(document, allow_nan=False).encode("utf-8")
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(describe_export_overflow(len(raw)))
    return gzip.compress(raw)


def import_results(raw):
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 256 MB limit.")
    try:
        if raw.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("Expanded file exceeds the 256 MB limit.")
        document = json.loads(raw)
        if (document.get("format") != "solar-balancer-results"
                or document.get("version") not in SUPPORTED_VERSIONS):
            raise ValueError("Unsupported result-file format or version.")
        kind = document["kind"]
        payload = _decode(document["payload"])
        if kind == "showcase":
            if not isinstance(payload, list):
                raise ValueError("Showcase results must be a list of locations.")
            required_metrics = ("average_annual_methane_kg", "lcom_usd_per_kg_ch4",
                "total_capex_usd", "annual_opex_usd_per_year", "plant_utilisation",
                "curtailment_fraction", "annual_balance_deficit_mwh",
                "storage_capacity_mwh", "forecast_cost_ratio")
            for row in payload:
                if not isinstance(row.get("site"), str) or not row["site"].strip():
                    raise ValueError("Each location needs a name.")
                if not (-90 <= row["lat"] <= 90 and -180 <= row["lon"] <= 180):
                    raise ValueError("Invalid location coordinates.")
                if not isinstance(row["data_status"], str):
                    raise ValueError("Missing location data status.")
                for key in required_metrics:
                    number = row[key]
                    if number is not None and (isinstance(number, bool)
                            or not isinstance(number, (float, int)) or not math.isfinite(number)):
                        raise ValueError(f"Invalid showcase metric: {key}.")
                if row["average_annual_methane_kg"] is None or row["average_annual_methane_kg"] < 0:
                    raise ValueError("Invalid methane output.")
        elif kind == "single_site":
            if not payload["results"] or not isinstance(payload["results"], dict):
                raise ValueError("No single-site results in file.")
            for result in payload["results"].values():
                if not isinstance(result, CaseResult) or not isinstance(result.perfect, SimulationResult):
                    raise ValueError("Invalid single-site case.")
                for simulation in (result.perfect, result.imperfect, result.imperfect_with_faults):
                    if simulation is not None and (not isinstance(simulation.hourly, pd.DataFrame)
                            or not isinstance(simulation.daily, pd.DataFrame)):
                        raise ValueError("Missing simulation tables.")
            if payload["information_mode"] not in {"perfect_only", "comparison"}:
                raise ValueError("Invalid information mode.")
            if not isinstance(payload["source_metadata"]["source"], str):
                raise ValueError("Missing source metadata.")
            if not isinstance(payload["evaluation_period"], str):
                raise ValueError("Missing evaluation period.")
        else:
            raise ValueError("Unknown result-file kind.")
        if kind == "showcase":
            payload = [normalize_showcase_ratio(row) for row in payload]
        cache = validate_weather_cache(_decode(document.get("weather_cache", [])))
        return kind, safe_inputs(payload), cache
    except (KeyError, TypeError, AttributeError, OSError, EOFError, UnicodeError,
            RecursionError) as exc:
        raise ValueError("Invalid or incomplete result file.") from exc
