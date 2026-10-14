#!/usr/bin/env python3
"""GFS operational maps for South America. Writes only under site/maps/gfs/."""
from __future__ import annotations
import json, os, re, traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
from sideral_map_style import render_map as render_sideral_map

EXTENT = (-90, -30, -60, 15)
HOURS = list(range(0, 145, 3)) + list(range(150, 241, 6))  # mesma cadência operacional do ECMWF
VARIANTS = {
    "0p25": "pgrb2.0p25",
    "0p50": "pgrb2.0p50",
    "1p00": "pgrb2.1p00",
}
FIELDS = {
    "temp2m": ("TMP:2 m", "Temperatura a 2 m", "°C", "coolwarm", -10, 40, "temp"),
    "dew2m": ("DPT:2 m", "Ponto de orvalho a 2 m", "°C", "BrBG", -20, 30, "temp"),
    "wind10m": (None, "Velocidade do vento a 10 m", "m/s", "turbo", 0, 30, "wind"),
    "precip": ("APCP:surface", "Precipitação acumulada", "mm", "turbo", 0, 80, "precip"),
    "mslp": ("PRMSL:mean sea level", "Pressão ao nível do mar", "hPa", "viridis", 940, 1040, "pressure"),
    "t850": ("TMP:850 mb", "Temperatura em 850 hPa", "°C", "coolwarm", -25, 25, "temp"),
    "gh500": ("HGT:500 mb", "Altura geopotencial em 500 hPa", "m", "viridis", 4800, 6000, "height"),
}
OUT = Path("site/maps/gfs")

def latest_run():
    # GFS 00/06/12/18Z. Lag by 8 hours to avoid selecting a cycle still publishing.
    t = datetime.now(timezone.utc) - timedelta(hours=8)
    return t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)

def extract(ds):
    candidates = []
    for name, da in ds.data_vars.items():
        arr = np.asarray(da.squeeze(drop=True).values)
        if arr.ndim == 2:
            candidates.append((name, da, arr))
    if not candidates:
        raise ValueError(f"Não achei grade 2D no GRIB; variáveis={list(ds.data_vars)}")
    name, da, values = candidates[0]
    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)
    values = np.asarray(values, dtype=float)
    # Most regular GFS fields have 1-D latitude and longitude coordinates.
    if lat.ndim == 1 and lon.ndim == 1 and values.shape == (lat.size, lon.size):
        lon = ((lon + 180) % 360) - 180
        ilon = np.argsort(lon)
        lon, values = lon[ilon], values[:, ilon]
        lat_mask = (lat >= EXTENT[2] - 2) & (lat <= EXTENT[3] + 2)
        lon_mask = (lon >= EXTENT[0] - 2) & (lon <= EXTENT[1] + 2)
        lat, values = lat[lat_mask], values[lat_mask, :]
        lon, values = lon[lon_mask], values[:, lon_mask]
        return lon, lat, values, str(da.attrs.get("units", ""))
    if lat.ndim == 2 and lon.ndim == 2:
        lon = ((lon + 180) % 360) - 180
        mask = (lon >= EXTENT[0]-2) & (lon <= EXTENT[1]+2) & (lat >= EXTENT[2]-2) & (lat <= EXTENT[3]+2)
        values = np.where(mask, values, np.nan)
        return lon, lat, values, str(da.attrs.get("units", ""))
    raise ValueError(f"Grade não suportada: values={values.shape}, lat={lat.shape}, lon={lon.shape}")

def make_map(lon, lat, values, units, key, variant, step, valid, out, wind_u=None, wind_v=None):
    _, label, unit, cmap, vmin, vmax, transform = FIELDS[key]
    title = f"{label}\nF{step:03d} Valid: {valid:%a %Y-%m-%d %HZ}"
    render_sideral_map(
        lon, lat, values, key, title, valid, out,
        source=f"NOAA/NCEP GFS {variant} Open Data", unit=units or unit,
        wind_u=wind_u, wind_v=wind_v,
    )

def load_one(h, search):
    result = h.xarray(search)
    datasets = result if isinstance(result, list) else [result]
    errors = []
    for ds in datasets:
        try:
            if not hasattr(ds, "data_vars"):
                errors.append(f"retorno não é Dataset: {type(ds).__name__}")
                continue
            return extract(ds)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                ds.close()
            except Exception:
                pass
    raise ValueError(f"Nenhum Dataset GRIB válido para {search!r}: {' | '.join(errors)}")

def main():
    run = latest_run()
    run_tag = run.strftime("%Y%m%d%H")
    report = {"model": "GFS", "run": run_tag, "source": "NOAA/NCEP via Herbie (AWS/NOMADS)",
              "extent": list(EXTENT), "forecastHours": HOURS, "variants": {}, "generated": [], "failures": [], "skipped": []}
    for variant, product in VARIANTS.items():
        variant_count = 0
        for hour in HOURS:
            try:
                h = Herbie(run.replace(tzinfo=None), model="gfs", product=product, fxx=hour, priority=["aws", "nomads"])
                for key, spec in FIELDS.items():
                    search, label, unit, cmap, vmin, vmax, transform = spec
                    if key == "precip" and hour == 0:
                        report["skipped"].append({"variant": variant, "hour": hour, "field": key, "reason": "F000 não tem precipitação acumulada desde o tempo inicial"})
                        continue
                    try:
                        wind_u = wind_v = None
                        if key == "wind10m":
                            lon, lat, wind_u, units = load_one(h, "UGRD:10 m")
                            lon2, lat2, wind_v, _ = load_one(h, "VGRD:10 m")
                            if not (np.allclose(lon, lon2) and np.allclose(lat, lat2) and wind_u.shape == wind_v.shape):
                                raise ValueError("Grades U/V do GFS não coincidem")
                            vals = np.hypot(wind_u, wind_v)
                        else:
                            lon, lat, vals, units = load_one(h, search)
                        out = OUT / variant / run_tag / key / f"f{hour:03d}.png"
                        make_map(lon, lat, vals, units, key, variant, hour, run + timedelta(hours=hour), out,
                                 wind_u=wind_u, wind_v=wind_v)
                        report["generated"].append(str(out)); variant_count += 1
                        print(f"OK {out}")
                    except Exception as exc:
                        report["failures"].append({"variant": variant, "hour": hour, "field": key,
                                                   "error": f"{type(exc).__name__}: {exc}"})
                        print(f"FALHA {variant} F{hour:03d} {key}: {exc}")
            except Exception as exc:
                report["failures"].append({"variant": variant, "hour": hour, "field": "cycle",
                                           "error": f"{type(exc).__name__}: {exc}"})
                print(f"FALHA ciclo GFS {variant} F{hour:03d}: {exc}")
        report["variants"][variant] = {"product": product, "mapsGenerated": variant_count}
    OUT.mkdir(parents=True, exist_ok=True)
    report["generatedCount"] = len(report["generated"])
    report["failureCount"] = len(report["failures"])
    (OUT / "generation-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("run","generatedCount","failureCount","variants")}, indent=2))
    if report["generatedCount"] == 0:
        raise SystemExit("Nenhum mapa GFS foi gerado; veja site/maps/gfs/generation-report.json")
if __name__ == "__main__":
    main()
