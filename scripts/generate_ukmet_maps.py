#!/usr/bin/env python3
"""UK Met Office Global deterministic native-grid bulletin runner.

Reads the public Met Office NetCDF objects from AWS S3 (N1280; 2560 x 1920 grid),
crops only by selecting native coordinates, and creates one South America map plus
five separate regional maps from the same unresampled native values.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import xarray as xr
from sideral_map_style import render_all_views, REGIONS

S3_BUCKET = "met-office-atmospheric-model-data"
S3_REGION = "eu-west-2"
S3_BASE = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/"
S3_PREFIX = "global-deterministic-10km"
OUT = Path("site/maps/ukmet")
CACHE = Path("work/ukmo-native")
EXTENT = (-90.0, -20.0, -60.0, 15.0)

# Met Office Global N1280 valid lead-time schedule (latest OS47 public files).
LONG_STEPS = list(range(0, 55)) + list(range(57, 145, 3)) + list(range(150, 169, 6))
SHORT_STEPS = list(range(0, 55)) + list(range(57, 70, 3))

FIELDS = {
    "temperature_2m": {
        "label": "Temperatura a 2 m", "unit": "°C", "level": None,
        "parameters": ["temperature_at_screen_level"],
    },
    "dew_point_2m": {
        "label": "Ponto de orvalho a 2 m", "unit": "°C", "level": None,
        "parameters": ["temperature_of_dew_point_at_screen_level"],
    },
    "wind_speed_10m": {
        "label": "Velocidade do vento a 10 m", "unit": "m s⁻¹", "level": None,
        "parameters": ["wind_speed_at_10m"],
    },
    "wind_gust_10m": {
        "label": "Rajada máxima a 10 m", "unit": "m s⁻¹", "level": None,
        "parameters": ["wind_gust_at_10m"],
    },
    "precipitation": {
        "label": "Precipitação acumulada no período", "unit": "mm", "level": None,
        "parameters": [],  # the accumulation period is selected for each forecast step
    },
    "pressure_msl": {
        "label": "Pressão ao nível do mar", "unit": "Pa", "level": None,
        "parameters": ["pressure_at_mean_sea_level"],
    },
    "cape_surface": {
        "label": "CAPE superficial", "unit": "J kg⁻¹", "level": None,
        "parameters": ["CAPE_surface", "cape_surface", "convective_available_potential_energy_at_surface"],
    },
    "cloud": {
        "label": "Cobertura total de nuvens", "unit": "%", "level": None,
        "parameters": [
            "cloud_amount_on_whole_atmosphere", "cloud_area_fraction_on_whole_atmosphere",
            "cloud_area_fraction", "cloud_amount_on_atmosphere",
        ],
    },
    "tcwv": {
        "label": "Água precipitável total", "unit": "kg m⁻²", "level": None,
        "parameters": [
            "total_column_integrated_water_vapour", "total_column_water_vapour",
            "atmosphere_mass_content_of_water_vapour", "atmosphere_mass_content_of_water_vapor",
        ],
    },
    "temperature_850hPa": {
        "label": "Temperatura em 850 hPa", "unit": "K", "level": 850,
        "parameters": ["air_temperature_on_pressure_levels"],
    },
    "wind_speed_850hPa": {
        "label": "Vento em 850 hPa", "unit": "m s⁻¹", "level": 850,
        "u_parameters": ["eastward_wind_on_pressure_levels"],
        "v_parameters": ["northward_wind_on_pressure_levels"],
    },
    "geopotential_height_500hPa": {
        "label": "Altura geopotencial em 500 hPa", "unit": "m", "level": 500,
        "parameters": ["geopotential_height_on_pressure_levels"],
    },
    "wind_speed_500hPa": {
        "label": "Vento em 500 hPa", "unit": "m s⁻¹", "level": 500,
        "u_parameters": ["eastward_wind_on_pressure_levels"],
        "v_parameters": ["northward_wind_on_pressure_levels"],
    },
    "wind_speed_250hPa": {
        "label": "Vento em 250 hPa", "unit": "m s⁻¹", "level": 250,
        "u_parameters": ["eastward_wind_on_pressure_levels"],
        "v_parameters": ["northward_wind_on_pressure_levels"],
    },
    "temperature_700hPa": {
        "label": "Temperatura em 700 hPa", "unit": "K", "level": 700,
        "parameters": ["air_temperature_on_pressure_levels"],
    },
    "temperature_500hPa": {
        "label": "Temperatura em 500 hPa", "unit": "K", "level": 500,
        "parameters": ["air_temperature_on_pressure_levels"],
    },
    "relative_humidity_850hPa": {
        "label": "Umidade relativa em 850 hPa", "unit": "%", "level": 850,
        "parameters": ["relative_humidity_on_pressure_levels"],
    },
    "wind_speed_700hPa": {
        "label": "Vento em 700 hPa", "unit": "m s⁻¹", "level": 700,
        "u_parameters": ["eastward_wind_on_pressure_levels"],
        "v_parameters": ["northward_wind_on_pressure_levels"],
    },
    "wind_speed_300hPa": {
        "label": "Vento em 300 hPa", "unit": "m s⁻¹", "level": 300,
        "u_parameters": ["eastward_wind_on_pressure_levels"],
        "v_parameters": ["northward_wind_on_pressure_levels"],
    },
    "geopotential_height_700hPa": {
        "label": "Altura geopotencial em 700 hPa", "unit": "m", "level": 700,
        "parameters": ["geopotential_height_on_pressure_levels"],
    },
    "geopotential_height_300hPa": {
        "label": "Altura geopotencial em 300 hPa", "unit": "m", "level": 300,
        "parameters": ["geopotential_height_on_pressure_levels"],
    },
}
REQUIRED_FIELDS = ("temperature_2m", "dew_point_2m", "wind_speed_10m", "pressure_msl", "precipitation", "cape_surface")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _request(url: str, method: str = "GET", headers: dict | None = None, timeout: int = 90):
    req = urllib.request.Request(url, method=method, headers={
        "User-Agent": "SideralMeteorologia-UKMO-native-bulletins/1.0", **(headers or {})
    })
    return urllib.request.urlopen(req, timeout=timeout)


def _object_url(key: str) -> str:
    return S3_BASE + urllib.parse.quote(key, safe="/-_.TZ")


def _prefix_for_run(run: dt.datetime) -> str:
    return f"{S3_PREFIX}/{run:%Y%m%d}T{run:%H}00Z/"


def _native_f000_exists(run: dt.datetime) -> bool:
    prefix = _prefix_for_run(run)
    key = f"{prefix}{run:%Y%m%d}T{run:%H}00Z-PT0000H00M-temperature_at_screen_level.nc"
    try:
        with _request(_object_url(key), method="HEAD", timeout=20):
            return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return False


def latest_run() -> dt.datetime:
    # UKMO Global files normally arrive ~3–6 hours after model initialisation;
    # probe the newest plausible cycle and fall back until an actual native file exists.
    now = dt.datetime.now(dt.timezone.utc)
    candidate = now - dt.timedelta(hours=7)
    cycle = candidate.replace(hour=(candidate.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for offset in range(20):
        run = cycle - dt.timedelta(hours=6 * offset)
        if _native_f000_exists(run):
            print(f"[UKMO native] rodada disponível confirmada no S3: {run:%Y-%m-%d %HZ}", flush=True)
            return run
        print(f"[UKMO native] rodada ainda não publicada: {run:%Y-%m-%d %HZ}", flush=True)
    raise RuntimeError("Nenhuma rodada UKMO nativa disponível no bucket público nas últimas 120 horas.")


def list_cycle_objects(run: dt.datetime) -> dict[int, dict[str, tuple[str, int]]]:
    """List metadata only for one cycle (about 3–5 pages); never downloads the whole forecast data."""
    prefix = _prefix_for_run(run)
    token = None
    catalogue: dict[int, dict[str, tuple[str, int]]] = {}
    pages = 0
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = S3_BASE + "?" + urllib.parse.urlencode(params)
        with _request(url, timeout=90) as response:
            xml_payload = response.read()
        root = ET.fromstring(xml_payload)
        pages += 1
        for obj in root.findall("s3:Contents", NS):
            key_node = obj.find("s3:Key", NS)
            size_node = obj.find("s3:Size", NS)
            if key_node is None or key_node.text is None:
                continue
            key = key_node.text
            name = key.rsplit("/", 1)[-1]
            match = re.fullmatch(r"(\d{8})T(\d{4})Z-PT(\d{4})H(\d{2})M-(.+)\.nc", name)
            if not match:
                continue
            lead = int(match.group(3)) + (int(match.group(4)) / 60.0)
            if abs(lead - round(lead)) > 1e-6:
                continue
            parameter = match.group(5).lower()
            size = int(size_node.text) if size_node is not None and size_node.text else 0
            catalogue.setdefault(int(round(lead)), {})[parameter] = (key, size)
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=NS).lower() == "true"
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=NS)
        if not truncated or not token:
            break
    if not catalogue:
        raise RuntimeError(f"O S3 não retornou arquivos NetCDF indexáveis para {prefix}")
    print(f"[UKMO native] catálogo de {sum(len(v) for v in catalogue.values())} arquivos; páginas={pages}", flush=True)
    return catalogue


def _find_param(catalogue, step: int, candidates: list[str]) -> tuple[str, str] | None:
    at_step = catalogue.get(step, {})
    for candidate in candidates:
        found = at_step.get(candidate.lower())
        if found:
            return found[0], candidate
    return None


def _precip_parameter(step: int) -> list[str]:
    period = 1 if step <= 54 else 3 if step <= 144 else 6
    return [
        f"precipitation_accumulation-PT{period:02d}H",
        f"precipitation_accumulation-PT{period:02d}h",
    ]


def _coord_name(ds: xr.Dataset, axis: str) -> str:
    aliases = {"latitude": {"latitude", "lat"}, "longitude": {"longitude", "lon", "long"}}
    for name, var in ds.variables.items():
        std = str(var.attrs.get("standard_name", "")).lower()
        lower = name.lower()
        if std == axis or lower in aliases[axis]:
            if getattr(var, "ndim", 0) == 1:
                return name
    raise ValueError(f"NetCDF sem coordenada 1D de {axis}: {list(ds.variables)[:30]}")


def _pressure_dimension(ds: xr.Dataset, da: xr.DataArray, target_hpa: int):
    candidates = []
    for dim in da.dims:
        if dim not in ds.variables:
            continue
        coord = ds[dim]
        if coord.ndim != 1 or coord.size < 2:
            continue
        name = (dim + " " + str(coord.attrs.get("standard_name", "")) + " " +
                str(coord.attrs.get("long_name", ""))).lower()
        if any(token in name for token in ("pressure", "isobaric", "level")):
            values = np.asarray(coord.values, dtype=float)
            finite = values[np.isfinite(values)]
            if not finite.size:
                continue
            target = target_hpa * 100.0 if np.nanmedian(finite) > 2000 else float(target_hpa)
            idx = int(np.nanargmin(np.abs(values - target)))
            candidates.append((dim, idx, float(values[idx]), values.size))
    if not candidates:
        return None, None
    dim, idx, selected, _ = sorted(candidates, key=lambda item: abs(item[2] - (target_hpa * 100 if item[2] > 2000 else target_hpa)))[0]
    return dim, idx


def read_native_subset(url: str, key: str, level_hpa: int | None = None):
    """Read just the South America native-grid chunks from a NetCDF object using HTTP byte ranges."""
    import fsspec
    cache_dir = CACHE / "http-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # blockcache issues HTTP Range requests instead of downloading the full 10 km global file.
    open_file = fsspec.open(url, mode="rb", block_size=2 * 1024 * 1024, cache_type="blockcache")
    with open_file as remote:
        with xr.open_dataset(remote, engine="h5netcdf", decode_times=False, cache=False) as ds:
            lat_name = _coord_name(ds, "latitude")
            lon_name = _coord_name(ds, "longitude")
            lat_coord = np.asarray(ds[lat_name].values, dtype=float)
            lon_coord_raw = np.asarray(ds[lon_name].values, dtype=float)
            lon_normalized = ((lon_coord_raw + 180.0) % 360.0) - 180.0
            lat_dim = ds[lat_name].dims[0]
            lon_dim = ds[lon_name].dims[0]
            y_idx = np.flatnonzero((lat_coord >= EXTENT[2] - 0.6) & (lat_coord <= EXTENT[3] + 0.6))
            x_idx = np.flatnonzero((lon_normalized >= EXTENT[0] - 0.6) & (lon_normalized <= EXTENT[1] + 0.6))
            if not len(y_idx) or not len(x_idx):
                raise ValueError(f"NetCDF sem células na área da América do Sul: lat={lat_coord.shape}, lon={lon_coord_raw.shape}")

            candidates = []
            for name, var in ds.data_vars.items():
                if name.lower() in {"flag", "crs", "latitude_bnds", "longitude_bnds"} or "flag" in name.lower():
                    continue
                if lat_dim not in var.dims or lon_dim not in var.dims:
                    continue
                if level_hpa is None and any(ds[d].size > 1 and any(t in d.lower() for t in ("pressure", "level")) for d in var.dims if d in ds.variables):
                    continue
                candidates.append((name, var))
            if not candidates:
                raise ValueError(f"NetCDF sem variável principal lat/lon para {key}: {list(ds.data_vars)}")
            var_name, var = candidates[0]
            lev_dim, lev_idx = (None, None)
            if level_hpa is not None:
                lev_dim, lev_idx = _pressure_dimension(ds, var, level_hpa)
                if lev_dim is None:
                    raise ValueError(f"{key}: não encontrei nível de pressão para {level_hpa} hPa em {var_name} ({var.dims})")

            # The South American longitudes occupy a contiguous native-grid interval on the 0–360 grid.
            y_slice = slice(int(y_idx.min()), int(y_idx.max()) + 1)
            x_slice = slice(int(x_idx.min()), int(x_idx.max()) + 1)
            indexers = {lat_dim: y_slice, lon_dim: x_slice}
            if lev_dim is not None:
                indexers[lev_dim] = lev_idx
            data_da = var.isel(indexers)
            order_dims = [lat_dim, lon_dim]
            for dim in list(data_da.dims):
                if dim not in order_dims and data_da.sizes[dim] == 1:
                    data_da = data_da.isel({dim: 0})
            if set(data_da.dims) != set(order_dims):
                extra = [dim for dim in data_da.dims if dim not in order_dims]
                raise ValueError(f"{key}: dimensões inesperadas após seleção: {data_da.dims} (extras={extra})")
            data = np.asarray(data_da.transpose(lat_dim, lon_dim).values, dtype=float)
            lat = lat_coord[y_slice]
            lon_unwrapped = lon_normalized[x_slice]
            order_x = np.argsort(lon_unwrapped)
            lon = lon_unwrapped[order_x]
            data = data[:, order_x]

            # Post-OS47 pressure-level files include a flag: 1 means below the surface.
            if lev_dim is not None and "flag" in ds.data_vars:
                try:
                    flag_da = ds["flag"]
                    flag_indexers = {lat_dim: y_slice, lon_dim: x_slice}
                    if lev_dim in flag_da.dims:
                        flag_indexers[lev_dim] = lev_idx
                    for dim in flag_da.dims:
                        if dim not in flag_indexers and flag_da.sizes[dim] == 1:
                            flag_indexers[dim] = 0
                    flag_slice = flag_da.isel({k: v for k, v in flag_indexers.items() if k in flag_da.dims})
                    flag_values = np.asarray(flag_slice.transpose(lat_dim, lon_dim).values)
                    flag_values = flag_values[:, order_x]
                    data = np.where(flag_values == 1, np.nan, data)
                except Exception as exc:
                    print(f"[UKMO native] aviso: máscara de pressão não aplicada em {key}: {exc}", flush=True)

            units = str(var.attrs.get("units", ""))
            if not np.isfinite(data).any():
                raise ValueError(f"{key}: todos os valores nativos da América do Sul são ausentes")
            return lon, lat, data, units


def _read_field(catalogue, step: int, field_key: str):
    spec = FIELDS[field_key]
    if field_key == "precipitation":
        candidates = _precip_parameter(step)
    else:
        candidates = spec.get("parameters", [])
    found = _find_param(catalogue, step, candidates)
    if not found:
        return None
    object_key, parameter = found
    url = _object_url(object_key)
    level = spec.get("level")
    lon, lat, values, unit = read_native_subset(url, field_key, level)
    return lon, lat, values, unit, parameter


def _read_wind_level(catalogue, step: int, field_key: str):
    spec = FIELDS[field_key]
    level = int(spec["level"])
    u_found = _find_param(catalogue, step, spec["u_parameters"])
    v_found = _find_param(catalogue, step, spec["v_parameters"])
    if not u_found or not v_found:
        return None
    lu, pu = u_found
    lv, pv = v_found
    u_lon, u_lat, u, u_unit = read_native_subset(_object_url(lu), field_key + "_u", level)
    v_lon, v_lat, v, v_unit = read_native_subset(_object_url(lv), field_key + "_v", level)
    if u.shape != v.shape or not np.allclose(u_lon, v_lon) or not np.allclose(u_lat, v_lat):
        raise ValueError(f"{field_key}: grades nativas U/V não coincidem")
    return u_lon, u_lat, np.hypot(u, v), u_unit or v_unit, u, v, f"{pu}+{pv}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    run = latest_run()
    tag = run.strftime("%Y%m%d%H")
    steps = LONG_STEPS if run.hour in (0, 12) else SHORT_STEPS
    catalogue = list_cycle_objects(run)
    report = {
        "model": "UK Met Office Global Deterministic",
        "modelId": "gl_det / N1280",
        "run": tag,
        "runUTC": run.isoformat(),
        "source": "Met Office Global Deterministic 10 km • public AWS NetCDF",
        "license": "British Crown copyright Met Office • CC BY-SA 4.0",
        "nativeResolution": {"grid": "N1280", "longitudePoints": 2560, "latitudePoints": 1920,
                             "spacingDegrees": {"longitude": 0.140625, "latitude": 0.09375}},
        "extent": list(EXTENT),
        "forecastHours": steps,
        "generated": [],
        "failures": [],
        "skipped": [],
        "fieldMapCounts": {},
    }
    for step in steps:
        valid = run + dt.timedelta(hours=step)
        for key, spec in FIELDS.items():
            out = OUT / tag / key / f"f{step:03d}.png"
            try:
                if key == "precipitation" and step == 0:
                    report["skipped"].append({"field": key, "step": step, "reason": "Sem acumulação de precipitação no F000"})
                    continue
                if "u_parameters" in spec and "v_parameters" in spec:
                    result = _read_wind_level(catalogue, step, key)
                    if result is None:
                        report["skipped"].append({"field": key, "step": step, "reason": "Parâmetro nativo não publicado nesta validade"})
                        continue
                    lon, lat, values, units, u, v, parameter = result
                    title = spec["label"]
                    written = render_all_views(
                        lon, lat, values, key, title, valid, out,
                        source="Met Office Global Deterministic N1280 • AWS open data (CC BY-SA 4.0)",
                        region_root=OUT / "regions", region_relative=out.relative_to(OUT),
                        unit=units or spec["unit"], wind_u=u, wind_v=v,
                        run_time=run, forecast_hour=step, model_label="UKMO Global 10 km • N1280",
                    )
                else:
                    result = _read_field(catalogue, step, key)
                    if result is None:
                        report["skipped"].append({"field": key, "step": step, "reason": "Parâmetro nativo não publicado nesta validade"})
                        continue
                    lon, lat, values, units, parameter = result
                    wind_direction = None
                    if key == "wind_speed_10m":
                        direction_field = _find_param(catalogue, step, ["wind_direction_at_10m"])
                        if direction_field:
                            direction_key, _ = direction_field
                            _, _, wind_direction, _ = read_native_subset(_object_url(direction_key), "wind_direction_10m")
                    title = spec["label"]
                    written = render_all_views(
                        lon, lat, values, key, title, valid, out,
                        source="Met Office Global Deterministic N1280 • AWS open data (CC BY-SA 4.0)",
                        region_root=OUT / "regions", region_relative=out.relative_to(OUT),
                        unit=units or spec["unit"], wind_direction=wind_direction,
                        run_time=run, forecast_hour=step, model_label="UKMO Global 10 km • N1280",
                    )
                report["generated"].extend(str(p) for p in written)
                report["fieldMapCounts"][key] = report["fieldMapCounts"].get(key, 0) + len(written)
                print(f"OK UKMO N1280 F{step:03d} {key}: {len(written)} PNGs", flush=True)
            except Exception as exc:
                report["failures"].append({
                    "field": key, "step": step,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"FALHA UKMO N1280 F{step:03d} {key}: {type(exc).__name__}: {exc}", flush=True)

    report["generatedCount"] = len(report["generated"])
    report["failureCount"] = len(report["failures"])
    report["regionalViewsPerSourceMap"] = 5
    report["gridResolutionPreserved"] = True
    (OUT / "generation-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "run": report["run"], "nativeResolution": report["nativeResolution"],
        "generatedCount": report["generatedCount"], "failureCount": report["failureCount"],
        "fields": report["fieldMapCounts"], "skippedCount": len(report["skipped"]),
    }, indent=2, ensure_ascii=False), flush=True)
    if report["generatedCount"] == 0:
        raise SystemExit("Nenhum mapa UKMO nativo válido foi gerado; verifique o relatório S3.")


if __name__ == "__main__":
    main()
