#!/usr/bin/env python3
"""UKMO Global 10 km maps using the same public Open-Meteo Single Runs source as Sideral Skew-T.

The Open-Meteo API returns point forecasts rather than native GRIB grids. This runner samples the
UKMO model on a regular 1-degree grid over South America, and records that sampling explicitly
in its report rather than claiming the PNGs are native 10-km grids.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sideral_map_style import render_map as render_sideral_map

BASE_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL_ID = "ukmo_global_deterministic_10km"
MODEL_LABEL = "UK Met Office Global 10 km"
USER_AGENT = "SideralMeteorologia/2.0 (UKMO map runner)"
EXTENT = (-90, -30, -60, 15)  # South America + adjacent oceans
LATITUDES = np.round(np.arange(-60.0, 15.001, 1.0), 3)
LONGITUDES = np.round(np.arange(-90.0, -29.999, 1.0), 3)
FORECAST_HOURS = list(range(0, 55)) + list(range(57, 145, 3)) + list(range(150, 169, 6))
API_FORECAST_HOURS = 169  # horizonte completo UKMO Global (7 dias), com a cadência nativa
BATCH_SIZE = int(os.environ.get("UKMET_BATCH_SIZE", "50"))
MAX_WORKERS = int(os.environ.get("UKMET_API_WORKERS", "3"))
OUT = Path("site/maps/ukmet")
RUNS_TO_TRY = 8

API_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "temperature_850hPa",
    "geopotential_height_500hPa",
    "wind_speed_10m",
    "wind_direction_10m",
]
FIELDS = {
    "temperature_2m": {"label": "Temperatura a 2 m", "unit": "°C", "cmap": "coolwarm"},
    "dew_point_2m": {"label": "Ponto de orvalho a 2 m", "unit": "°C", "cmap": "BrBG"},
    "precipitation": {"label": "Precipitação horária", "unit": "mm", "cmap": "YlGnBu"},
    "pressure_msl": {"label": "Pressão ao nível do mar", "unit": "hPa", "cmap": "viridis"},
    "temperature_850hPa": {"label": "Temperatura em 850 hPa", "unit": "°C", "cmap": "coolwarm"},
    "geopotential_height_500hPa": {"label": "Altura geopotencial em 500 hPa", "unit": "m", "cmap": "viridis"},
    "wind_speed_10m": {"label": "Velocidade do vento a 10 m", "unit": "kn", "cmap": "viridis"},
}


def api_json(params: dict, retries: int = 5):
    url = BASE_URL + "?" + urlencode(params)
    last_error = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(str(payload.get("reason", "Open-Meteo returned an API error")))
            return payload
        except HTTPError as exc:
            detail = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
                detail = body[:500]
            except Exception:
                pass
            last_error = f"HTTP {exc.code} {detail}"
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == retries:
                break
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 == retries:
                break
        pause = min(30, 2 ** attempt)
        print(f"[Open-Meteo] tentativa {attempt + 1}/{retries}: {last_error}; nova tentativa em {pause}s", flush=True)
        time.sleep(pause)
    raise RuntimeError(f"Open-Meteo Single Runs indisponível após {retries} tentativas: {last_error}")


def candidate_runs(now: dt.datetime):
    current_hour = (now.hour // 6) * 6
    latest = now.replace(hour=current_hour, minute=0, second=0, microsecond=0)
    for back in range(RUNS_TO_TRY):
        yield latest - dt.timedelta(hours=6 * back)


def make_params(run: dt.datetime, points: list[tuple[float, float]]):
    return {
        "latitude": ",".join(f"{lat:.3f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.3f}" for _, lon in points),
        "hourly": ",".join(API_VARIABLES),
        "models": MODEL_ID,
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": str(API_FORECAST_HOURS),
        "wind_speed_unit": "kn",
        "temperature_unit": "celsius",
        "pressure_unit": "hPa",
        "precipitation_unit": "mm",
        "timeformat": "iso8601",
        "timezone": "UTC",
        "cell_selection": "nearest",
    }


def payload_locations(payload, expected_count: int):
    if isinstance(payload, dict):
        locations = [payload]
    elif isinstance(payload, list):
        locations = payload
    else:
        raise RuntimeError(f"Resposta Open-Meteo inesperada: {type(payload).__name__}")
    if len(locations) != expected_count:
        raise RuntimeError(f"Open-Meteo retornou {len(locations)} pontos; esperados {expected_count}")
    return locations


def find_available_run():
    now = dt.datetime.now(dt.timezone.utc)
    probe = [(float(LATITUDES[len(LATITUDES) // 2]), float(LONGITUDES[len(LONGITUDES) // 2]))]
    errors = []
    for run in candidate_runs(now):
        try:
            payload = api_json(make_params(run, probe), retries=2)
            loc = payload_locations(payload, 1)[0]
            hourly = loc.get("hourly") or {}
            times = hourly.get("time") or []
            if not times or not any(hourly.get(key) for key in ("temperature_2m", "pressure_msl", "precipitation")):
                raise RuntimeError("resposta sem séries horárias úteis")
            print(f"[UKMO] rodada confirmada pela API: {run:%Y-%m-%d %HZ}", flush=True)
            return run
        except Exception as exc:
            note = f"{run:%Y-%m-%d %HZ}: {exc}"
            errors.append(note)
            print(f"[UKMO] rodada ainda indisponível: {note}", flush=True)
    raise RuntimeError("Nenhuma rodada UKMO recente ficou disponível na Single Runs API: " + " | ".join(errors))


def request_batch(args):
    batch_number, run, points = args
    payload = api_json(make_params(run, points))
    locations = payload_locations(payload, len(points))
    records = []
    for point, location in zip(points, locations):
        hourly = location.get("hourly") or {}
        if not hourly.get("time"):
            raise RuntimeError(f"Ponto {point} sem eixo temporal na resposta")
        record = {"point": point, "time": hourly["time"], "values": {}}
        for variable in API_VARIABLES:
            raw = hourly.get(variable)
            if not isinstance(raw, list):
                record["values"][variable] = []
            else:
                record["values"][variable] = [
                    float(value) if value is not None and np.isfinite(float(value)) else np.nan
                    for value in raw
                ]
        records.append(record)
    print(f"[Open-Meteo] lote {batch_number}: {len(records)} pontos", flush=True)
    return records


def collect_grid(run: dt.datetime):
    points = [(float(lat), float(lon)) for lat in LATITUDES for lon in LONGITUDES]
    batches = [points[i:i + BATCH_SIZE] for i in range(0, len(points), BATCH_SIZE)]
    values = {
        key: np.full((len(LATITUDES), len(LONGITUDES), API_FORECAST_HOURS), np.nan, dtype=np.float32)
        for key in API_VARIABLES
    }
    time_axis = None
    jobs = [(i + 1, run, batch) for i, batch in enumerate(batches)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(request_batch, job): job[0] for job in jobs}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            records = future.result()
            done += 1
            for record in records:
                lat, lon = record["point"]
                yi = int(round(lat - LATITUDES[0]))
                xi = int(round(lon - LONGITUDES[0]))
                if time_axis is None:
                    time_axis = record["time"]
                elif record["time"] != time_axis:
                    raise RuntimeError("Eixos temporais divergentes entre pontos UKMO")
                for key, series in record["values"].items():
                    count = min(len(series), API_FORECAST_HOURS)
                    if count:
                        values[key][yi, xi, :count] = np.asarray(series[:count], dtype=np.float32)
            if done % 10 == 0 or done == len(batches):
                print(f"[Open-Meteo] lotes concluídos: {done}/{len(batches)}", flush=True)
    if not time_axis:
        raise RuntimeError("A API não retornou eixo temporal para a grade UKMO")
    return values, time_axis



def main():
    if BATCH_SIZE < 2 or BATCH_SIZE > 80:
        raise SystemExit("UKMET_BATCH_SIZE deve ficar entre 2 e 80.")
    if MAX_WORKERS < 1 or MAX_WORKERS > 5:
        raise SystemExit("UKMET_API_WORKERS deve ficar entre 1 e 5.")
    OUT.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc)
    report = {
        "model": MODEL_LABEL,
        "modelId": MODEL_ID,
        "source": "Open-Meteo Single Runs API (the same source/model ID used by Sideral Skew-T)",
        "createdUTC": created.isoformat(),
        "status": "pending",
        "extent": list(EXTENT),
        "gridSampling": {
            "latitudeStepDegrees": 1.0,
            "longitudeStepDegrees": 1.0,
            "note": "Valores UKMO consultados em pontos regulares pela API; PNG não é um GRIB nativo de 10 km.",
        },
        "forecastHours": FORECAST_HOURS,
        "nativeCadence": {"hourlyThrough": 54, "threeHourlyThrough": 144, "sixHourlyThrough": 168},
        "generated": [],
        "failures": [],
    }
    try:
        run = find_available_run()
        values, time_axis = collect_grid(run)
        report["run"] = run.strftime("%Y%m%d%H")
        report["timeAxisStartUTC"] = time_axis[0]
        report["gridPoints"] = int(len(LATITUDES) * len(LONGITUDES))
        report["validPointsByField"] = {
            key: int(np.isfinite(arr[:, :, 0]).sum()) for key, arr in values.items()
        }
        wind_speed = values["wind_speed_10m"]
        wind_direction = values["wind_direction_10m"]
        for field_key in FIELDS:
            for step in FORECAST_HOURS:
                if step >= len(time_axis):
                    report["failures"].append({"field": field_key, "step": step, "error": "validade fora da resposta API"})
                    continue
                if field_key == "precipitation" and step == 0:
                    report.setdefault("skipped", []).append({
                        "field": field_key, "step": step,
                        "reason": "F000 não representa precipitação acumulada desde o tempo inicial",
                    })
                    continue
                valid_time = str(time_axis[step])
                out = OUT / report["run"] / field_key / f"f{step:03d}.png"
                try:
                    wind_dir = wind_direction[:, :, step] if field_key == "wind_speed_10m" else None
                    unit = FIELDS[field_key]["unit"]
                    title = f"{FIELDS[field_key]['label']}\nF{step:03d} Valid: {valid_time.replace('T', ' ')}Z"
                    render_sideral_map(
                        LONGITUDES, LATITUDES, values[field_key][:, :, step],
                        field_key, title, valid_time, out,
                        source="UK Met Office Global 10 km via Open-Meteo Single Runs",
                        unit=unit, wind_direction=wind_dir,
                    )
                    report["generated"].append(str(out))
                    print(f"OK {out}", flush=True)
                except Exception as exc:
                    report["failures"].append({
                        "field": field_key, "step": step,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"FALHA {field_key} F{step:03d}: {exc}", flush=True)
        report["generatedCount"] = len(report["generated"])
        report["failureCount"] = len(report["failures"])
        if report["generatedCount"] == 0:
            report["status"] = "failed"
        elif report["failureCount"]:
            report["status"] = "complete_with_errors"
        else:
            report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["fatalError"] = f"{type(exc).__name__}: {exc}"
        report["generatedCount"] = len(report["generated"])
        report["failureCount"] = len(report["failures"]) + 1
    (OUT / "generation-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        key: report.get(key) for key in (
            "status", "run", "gridPoints", "generatedCount", "failureCount", "fatalError"
        )
    }, indent=2, ensure_ascii=False), flush=True)
    if report["status"] == "failed":
        raise SystemExit("UKMO map generation failed; inspect site/maps/ukmet/generation-report.json")


if __name__ == "__main__":
    main()
