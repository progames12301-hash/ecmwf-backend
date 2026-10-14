"""Sideral bulletin renderer: native-grid PNGs, white-based palettes, political divisions and regional crops."""
from __future__ import annotations

import io
import json
import os
import time
import urllib.request
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature

EXTENT_SOUTH_AMERICA = (-90.0, -20.0, -60.0, 15.0)
REGIONS = {
    "nordeste": {"label": "Nordeste", "extent": (-49.5, -32.0, -18.5, 1.5)},
    "sudeste": {"label": "Sudeste", "extent": (-54.5, -39.0, -26.5, -14.0)},
    "sul": {"label": "Sul", "extent": (-58.5, -47.0, -34.5, -22.0)},
    "norte": {"label": "Norte", "extent": (-75.5, -43.0, -13.5, 6.0)},
    "centro-oeste": {"label": "Centro-Oeste", "extent": (-62.5, -46.0, -25.5, -7.0)},
}

# Palettes intentionally start at white, including CAPE and cloud cover.
CMAPS = {
    "precip": LinearSegmentedColormap.from_list("sideral_precip", [
        "#ffffff", "#ededed", "#bcbcbc", "#777777", "#5140bf", "#3437de",
        "#3866ed", "#258fe9", "#29c7e6", "#59d7c5", "#40c65e", "#b4db3a",
        "#f5e936", "#ffc21b", "#ff861b", "#f23825", "#c800b9", "#771078",
    ], N=256),
    "temperature": LinearSegmentedColormap.from_list("sideral_temperature", [
        "#ffffff", "#d8f0ff", "#91d7f4", "#3b90db", "#3155c8",
        "#2cb8cc", "#54d68e", "#e8ed5c", "#ffb13b", "#ed4c32", "#9f071a",
    ], N=256),
    "wind": LinearSegmentedColormap.from_list("sideral_wind", [
        "#ffffff", "#c8f1f7", "#66d4e7", "#2c8fe2", "#3449c7",
        "#2fbf78", "#c6dc43", "#f3e638", "#ff9b24", "#e62a26", "#771078",
    ], N=256),
    "pressure": LinearSegmentedColormap.from_list("sideral_pressure", [
        "#ffffff", "#d9efff", "#94cfef", "#3b92d8", "#3756be",
        "#40b7b8", "#9ed56a", "#f2df61", "#ef9a40", "#bc433c",
    ], N=256),
    "height": LinearSegmentedColormap.from_list("sideral_height", [
        "#ffffff", "#c6e6f5", "#68afd6", "#3e7db8", "#5149a3",
        "#47a79e", "#a8cc70", "#f0dc67", "#e99b4c",
    ], N=256),
    "cloud": LinearSegmentedColormap.from_list("sideral_cloud", [
        "#ffffff", "#f0f0f0", "#d8d8d8", "#b9b9b9", "#929292", "#696969", "#383838",
    ], N=256),
    "tcwv": LinearSegmentedColormap.from_list("sideral_tcwv", [
        "#ffffff", "#d7f2ff", "#8cdaf2", "#309dd8", "#2e69bd",
        "#23b6af", "#54c76d", "#c8dc55", "#f2df57",
    ], N=256),
    "cape": LinearSegmentedColormap.from_list("sideral_cape", [
        "#ffffff", "#f0f0f0", "#d8f0c1", "#a4d96c", "#43b66b",
        "#fff04a", "#ffb52b", "#f05a24", "#d7191c", "#a50f79", "#5b167a",
    ], N=256),
    "generic": LinearSegmentedColormap.from_list("sideral_generic", [
        "#ffffff", "#d8eeff", "#80c8e9", "#3985c7", "#49b89a",
        "#b9d950", "#f8df4a", "#ef983b", "#cf3535",
    ], N=256),
}
NE_ADMIN1_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson"
SIDERAL_LOGO_URL = "https://raw.githubusercontent.com/progames12301-hash/sideral-backend/main/assets/sideral-logo.png"
ASSET_DIR = Path(os.environ.get("SIDERAL_MAP_ASSET_DIR", "work/shared-map-assets"))
ADMIN1_FILE = ASSET_DIR / "ne_50m_admin_1_states_provinces.geojson"
LOGO_FILE = ASSET_DIR / "sideral-logo.png"


def _fetch_to_cache(url: str, path: Path, min_size: int = 1024) -> bool:
    try:
        if path.exists() and path.stat().st_size >= min_size:
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".download")
        req = urllib.request.Request(url, headers={"User-Agent": "SideralMeteorologia-bulletin/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(tmp, "wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
        if tmp.stat().st_size < min_size:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
        return True
    except Exception as exc:
        print(f"[map-style] asset indisponível ({url}): {type(exc).__name__}: {exc}", flush=True)
        try:
            path.with_suffix(path.suffix + ".download").unlink(missing_ok=True)
        except OSError:
            pass
        return False


@lru_cache(maxsize=1)
def admin1_geometries():
    """Load first-order state/province boundaries from Natural Earth once per runner process."""
    try:
        from shapely.geometry import box, shape
        if not _fetch_to_cache(NE_ADMIN1_URL, ADMIN1_FILE, min_size=100_000):
            return ()
        with open(ADMIN1_FILE, "r", encoding="utf-8") as stream:
            collection = json.load(stream)
        geometries = []
        for feature in collection.get("features", []):
            raw = feature.get("geometry")
            if not raw:
                continue
            geom = shape(raw)
            if not geom.is_empty and geom.bounds[0] <= -19 and geom.bounds[2] >= -91 and geom.bounds[1] <= 16 and geom.bounds[3] >= -61:
                geometries.append(geom.boundary)
        print(f"[map-style] limites administrativos carregados: {len(geometries)}", flush=True)
        return tuple(geometries)
    except Exception as exc:
        print(f"[map-style] aviso: limites de estados/províncias não carregados: {type(exc).__name__}: {exc}", flush=True)
        return ()


@lru_cache(maxsize=1)
def logo_image():
    try:
        if not _fetch_to_cache(SIDERAL_LOGO_URL, LOGO_FILE, min_size=100):
            return None
        return plt.imread(str(LOGO_FILE))
    except Exception as exc:
        print(f"[map-style] aviso: logo Sideral indisponível: {type(exc).__name__}: {exc}", flush=True)
        return None


def _as_datetime(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y%m%d%H", "%Y-%m-%d %HZ", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(str(value), fmt)
            except ValueError:
                continue
    return None


def _family(key: str, title: str = "") -> str:
    k = str(key).lower()
    t = str(title).lower()
    if any(x in k for x in ("precip", "apcp", "rainfall", "tot_prec")) or k in {"tp", "precip_rate"}:
        return "precip"
    if "cape" in k:
        return "cape"
    if any(x in k for x in ("cloud", "tcc", "clct")):
        return "cloud"
    if any(x in k for x in ("tcwv", "water_vapour", "water_vapor", "pwat", "tqv", "precipitable")):
        return "tcwv"
    if any(x in k for x in ("pressure", "mslp", "prmsl")):
        return "pressure"
    if any(x in k for x in ("height", "geopotential", "gh500", "hgt")):
        return "height"
    if any(x in k for x in ("wind", "gust")) or k.startswith(("u850", "v850", "u500", "v500", "u250", "v250")):
        return "wind"
    if any(x in k for x in ("temp", "dew", "2t", "2d", "t850")):
        return "temperature"
    if "taxa" in t or "rate" in t:
        return "precip"
    return "generic"


def _levels_and_norm(family: str, key: str, title: str):
    from matplotlib.colors import BoundaryNorm
    if family == "precip":
        if "taxa" in str(title).lower() or "rate" in str(key).lower():
            ticks = np.array([0, .1, .25, .5, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 75], dtype=float)
            vmax = 75.0
        elif "período" in str(title).lower() or "period" in str(title).lower():
            ticks = np.array([0, .1, .25, .5, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 75], dtype=float)
            vmax = 75.0
        else:
            ticks = np.array([0, .1, .25, .5, 1, 2.5, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 500], dtype=float)
            vmax = 500.0
        return Normalize(0.0, vmax, clip=True), ticks
    fixed = {
        "temperature": (-10.0, 40.0, np.arange(-10, 41, 5)),
        "wind": (0.0, 120.0 if any(x in str(key).lower() for x in ("850", "500", "250")) else 80.0, None),
        "pressure": (960.0, 1040.0, np.arange(960, 1041, 4)),
        "height": (4800.0, 6000.0, np.arange(4800, 6001, 60)),
        "cloud": (0.0, 100.0, np.arange(0, 101, 10)),
        "tcwv": (0.0, 70.0, np.arange(0, 71, 5)),
        "cape": (0.0, 6000.0, np.array([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000])),
        "generic": (0.0, 1.0, None),
    }
    lo, hi, ticks = fixed.get(family, fixed["generic"])
    if family == "temperature" and any(x in str(key).lower() for x in ("850", "500", "250")):
        lo, hi, ticks = (-80.0, 30.0, np.arange(-80, 31, 10))
    if family == "height" and "250" in str(key).lower():
        lo, hi, ticks = (9000.0, 12000.0, np.arange(9000, 12001, 200))
    return Normalize(lo, hi, clip=True), ticks


def _add_admin1(ax, extent):
    geoms = admin1_geometries()
    if geoms:
        from shapely.geometry import box
        # Keep only polygons whose bounds intersect the current view before handing them to Cartopy.
        xmin, xmax, ymin, ymax = extent
        chosen = []
        for geom in geoms:
            bounds = geom.bounds
            if bounds[0] <= xmax and bounds[2] >= xmin and bounds[1] <= ymax and bounds[3] >= ymin:
                chosen.append(geom)
        if chosen:
            ax.add_geometries(chosen, ccrs.PlateCarree(), facecolor="none",
                              edgecolor="#191919", linewidth=0.48 if extent == EXTENT_SOUTH_AMERICA else 0.58,
                              zorder=6)
    else:
        print("[map-style] aviso: mapa sem limites internos administrativos", flush=True)


def _crop_native(lon, lat, data, extent, wind_u=None, wind_v=None, wind_direction=None, unstructured=False):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    data = np.asarray(data, dtype=float).squeeze()
    u = None if wind_u is None else np.asarray(wind_u, dtype=float).squeeze()
    v = None if wind_v is None else np.asarray(wind_v, dtype=float).squeeze()
    direction = None if wind_direction is None else np.asarray(wind_direction, dtype=float).squeeze()
    xmin, xmax, ymin, ymax = extent

    if unstructured or (data.ndim == 1 and lon.ndim == lat.ndim == 1 and lon.size == lat.size == data.size):
        if lon.ndim != 1 or lat.ndim != 1 or data.ndim != 1 or lon.size != lat.size or lat.size != data.size:
            raise ValueError(f"Grade dispersa inválida: lon={lon.shape}, lat={lat.shape}, data={data.shape}")
        keep = (lon >= xmin - 0.35) & (lon <= xmax + 0.35) & (lat >= ymin - 0.35) & (lat <= ymax + 0.35)
        if int(keep.sum()) < 20:
            raise ValueError(f"Poucos pontos nativos na região: {int(keep.sum())}")
        return lon[keep], lat[keep], data[keep], (
            u[keep] if u is not None and u.shape == keep.shape else u
        ), (
            v[keep] if v is not None and v.shape == keep.shape else v
        ), (
            direction[keep] if direction is not None and direction.shape == keep.shape else direction
        ), True

    if lon.ndim == lat.ndim == 1:
        if data.ndim != 2 or data.shape != (lat.size, lon.size):
            raise ValueError(f"Grade regular incompatível: values={data.shape}, lat={lat.shape}, lon={lon.shape}")
        ix = np.flatnonzero((lon >= xmin - 0.5) & (lon <= xmax + 0.5))
        iy = np.flatnonzero((lat >= ymin - 0.5) & (lat <= ymax + 0.5))
        if not len(ix) or not len(iy):
            raise ValueError(f"Grade sem interseção com a extensão {extent}")
        sl = np.ix_(iy, ix)
        data = data[sl]
        if u is not None and u.ndim == 2:
            u = u[sl]
        if v is not None and v.ndim == 2:
            v = v[sl]
        if direction is not None and direction.ndim == 2:
            direction = direction[sl]
        return lon[ix], lat[iy], data, u, v, direction, False

    if lon.ndim == lat.ndim == 2 and data.shape == lon.shape == lat.shape:
        inside = ((lon >= xmin - 0.5) & (lon <= xmax + 0.5)
                  & (lat >= ymin - 0.5) & (lat <= ymax + 0.5))
        iy = np.flatnonzero(inside.any(axis=1))
        ix = np.flatnonzero(inside.any(axis=0))
        if not len(ix) or not len(iy):
            raise ValueError(f"Grade curvilínea sem interseção com a extensão {extent}")
        sl = np.ix_(iy, ix)
        lon, lat, data = lon[sl], lat[sl], data[sl]
        if u is not None and u.ndim == 2:
            u = u[sl]
        if v is not None and v.ndim == 2:
            v = v[sl]
        if direction is not None and direction.ndim == 2:
            direction = direction[sl]
        return lon, lat, data, u, v, direction, False
    raise ValueError(f"Grade não suportada: values={data.shape}, lat={lat.shape}, lon={lon.shape}")


def render_map(lon, lat, values, key, title, valid, output, source,
               unit="", wind_u=None, wind_v=None, wind_direction=None,
               unstructured=False, extent=None, region_name=None,
               run_time=None, forecast_hour=None, model_label=None):
    """Render a map using native source values; regional products are native-grid subsets, never resized forecasts."""
    extent = tuple(extent or EXTENT_SOUTH_AMERICA)
    valid_dt = _as_datetime(valid)
    run_dt = _as_datetime(run_time)
    if run_dt is None and valid_dt is not None and forecast_hour is not None:
        run_dt = valid_dt - timedelta(hours=int(forecast_hour))
    data_family = _family(key, title)
    source_unit = str(unit or "").strip().lower()
    data = np.asarray(values, dtype=float).squeeze().copy()
    # Derive wind components before spatial cropping, so direction/speed and barbs use the same native cells.
    if data_family == "wind" and wind_direction is not None and wind_u is None and wind_v is None:
        direction_native = np.deg2rad(np.asarray(wind_direction, dtype=float).squeeze())
        wind_u = -data * np.sin(direction_native)
        wind_v = -data * np.cos(direction_native)
    lon, lat, data, wind_u, wind_v, wind_direction, scattered = _crop_native(
        lon, lat, data, extent, wind_u, wind_v, wind_direction, unstructured
    )
    data[~np.isfinite(data)] = np.nan
    out_unit = unit or ""

    if data_family == "temperature":
        if np.isfinite(data).any() and float(np.nanmedian(data)) > 150:
            data -= 273.15
        out_unit = "°C"
    elif data_family == "pressure":
        if np.isfinite(data).any() and float(np.nanmedian(data)) > 2000:
            data /= 100.0
        out_unit = "hPa"
    elif data_family == "height":
        if np.isfinite(data).any() and float(np.nanmedian(np.abs(data))) > 50000:
            data /= 9.80665
        out_unit = "m"
    elif data_family == "cloud":
        if np.isfinite(data).any() and float(np.nanpercentile(data, 95)) <= 1.01:
            data *= 100.0
        data = np.clip(data, 0, 100)
        out_unit = "%"
    elif data_family == "precip":
        data = np.maximum(data, 0)
        out_unit = "mm h⁻¹" if "taxa" in str(title).lower() or "rate" in str(key).lower() else "mm"
    elif data_family == "tcwv":
        data = np.maximum(data, 0)
        out_unit = unit or "kg m⁻²"
    elif data_family == "cape":
        data = np.maximum(data, 0)
        out_unit = unit or "J kg⁻¹"

    if data_family == "wind":
        factor = 1.0
        if "knot" in source_unit or "kn" == source_unit:
            factor = 1.852
        elif source_unit in ("m/s", "m s-1", "m s⁻¹", "ms-1") or "m s" in source_unit:
            factor = 3.6
        data *= factor
        out_unit = "km h⁻¹"

    norm, ticks = _levels_and_norm(data_family, key, title)
    cmap = CMAPS.get(data_family, CMAPS["generic"])
    fig_w, fig_h = (14.0, 8.2) if region_name is None else (10.5, 8.2)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=130, facecolor="white")
    ax = fig.add_axes([0.035, 0.14, 0.93, 0.78])
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="#aaaaaa", linewidth=.25, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=.55, edgecolor="#191919", zorder=5)
    ax.coastlines(resolution="110m", linewidth=.7, color="#191919", zorder=5)

    mappable = None
    mask_data = np.ma.masked_invalid(data)
    if scattered:
        tri = mtri.Triangulation(lon, lat)
        triangles = tri.triangles
        x, y = lon[triangles], lat[triangles]
        max_edge = np.maximum.reduce([
            np.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]),
            np.hypot(x[:, 1] - x[:, 2], y[:, 1] - y[:, 2]),
            np.hypot(x[:, 2] - x[:, 0], y[:, 2] - y[:, 0]),
        ])
        tri.set_mask((max_edge > 2.0) | np.any(~np.isfinite(data[triangles]), axis=1))
        face_values = np.mean(data[triangles], axis=1)
        face_values = np.ma.masked_invalid(face_values)
        mappable = ax.tripcolor(tri, facecolors=face_values, shading="flat", cmap=cmap, norm=norm, zorder=2)
    elif lon.ndim == lat.ndim == 1:
        mappable = ax.pcolormesh(lon, lat, mask_data, transform=ccrs.PlateCarree(),
                                 shading="auto", cmap=cmap, norm=norm, rasterized=True, zorder=2)
    else:
        mappable = ax.pcolormesh(lon, lat, mask_data, transform=ccrs.PlateCarree(),
                                 shading="auto", cmap=cmap, norm=norm, rasterized=True, zorder=2)

    if data_family == "pressure" and not scattered and lon.ndim == lat.ndim == 1 and data.ndim == 2:
        try:
            levels = np.arange(960, 1041, 4)
            contours = ax.contour(lon, lat, data, levels=levels, colors="#202020",
                                  linewidths=.6, transform=ccrs.PlateCarree(), zorder=4)
            ax.clabel(contours, fmt="%d", fontsize=7, inline=True)
        except Exception as exc:
            print(f"[map-style] isóbaras omitidas em {key}: {exc}", flush=True)

    if data_family == "height" and not scattered and lon.ndim == lat.ndim == 1 and data.ndim == 2:
        try:
            finite = data[np.isfinite(data)]
            if finite.size:
                lo, hi = float(np.nanpercentile(finite, 3)), float(np.nanpercentile(finite, 97))
                gap = 60.0 if np.nanmedian(finite) > 3000 else 30.0
                levels = np.arange(np.floor(lo / gap) * gap, np.ceil(hi / gap) * gap + gap, gap)
                contours = ax.contour(lon, lat, data, levels=levels, colors="#202020",
                                       linewidths=.45, transform=ccrs.PlateCarree(), zorder=4)
                ax.clabel(contours, fmt="%d", fontsize=6)
        except Exception as exc:
            print(f"[map-style] isoípsas omitidas em {key}: {exc}", flush=True)

    if data_family == "wind" and wind_u is not None and wind_v is not None:
        try:
            uu, vv = np.asarray(wind_u, dtype=float), np.asarray(wind_v, dtype=float)
            if uu.shape == data.shape and vv.shape == data.shape:
                if scattered:
                    stride = max(1, lon.size // 650)
                    ax.barbs(lon[::stride], lat[::stride], uu[::stride], vv[::stride],
                             length=4.4, linewidth=.35, transform=ccrs.PlateCarree(), zorder=7)
                elif lon.ndim == lat.ndim == 1:
                    stride_x, stride_y = max(1, lon.size // 35), max(1, lat.size // 24)
                    ax.barbs(lon[::stride_x], lat[::stride_y], uu[::stride_y, ::stride_x], vv[::stride_y, ::stride_x],
                             length=5, linewidth=.35, transform=ccrs.PlateCarree(), zorder=7)
                else:
                    stride = max(1, data.shape[1] // 35, data.shape[0] // 24)
                    ax.barbs(lon[::stride, ::stride], lat[::stride, ::stride], uu[::stride, ::stride], vv[::stride, ::stride],
                             length=5, linewidth=.35, transform=ccrs.PlateCarree(), zorder=7)
        except Exception as exc:
            print(f"[map-style] barbelas omitidas em {key}: {exc}", flush=True)

    _add_admin1(ax, extent)
    gl = ax.gridlines(draw_labels=True, linewidth=.28, color="#777777", alpha=.45, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    model_text = model_label or source
    model_text = str(model_text).replace("Open Data", "").strip(" •")
    if run_dt is not None:
        run_text = run_dt.strftime("%d/%m/%Y %HZ")
    else:
        run_text = "rodada não informada"
    if valid_dt is not None:
        valid_text = valid_dt.strftime("%d/%m/%y %HZ")
    else:
        valid_text = str(valid or "validade não informada")
    step = int(forecast_hour) if forecast_hour is not None else (
        int((valid_dt - run_dt).total_seconds() // 3600) if valid_dt is not None and run_dt is not None else None
    )
    step_text = f"(+{step:03d}h)" if step is not None else ""
    fig.text(.014, .983, f"Rodada: {run_text}", ha="left", va="top", fontsize=9, color="#111111")
    fig.text(.5, .983, f"Modelo: {model_text}", ha="center", va="top", fontsize=9, color="#111111")
    fig.text(.986, .983, f"Previsão: {valid_text} {step_text}", ha="right", va="top", fontsize=9, color="#111111")
    subtitle = str(title).replace("\n", " ").strip()
    if region_name:
        subtitle = f"{subtitle} — {region_name.upper()}"
    fig.text(.5, .952, subtitle, ha="center", va="top", fontsize=10, color="#343434")

    cax = fig.add_axes([.10, .078, .80, .018])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(out_unit, fontsize=8, labelpad=1)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    if ticks is not None and len(ticks) <= 21:
        cbar.set_ticks(ticks)

    fig.text(.014, .023, f"Fonte: {source} • Limites internos: Natural Earth Admin-1", ha="left", va="bottom",
             fontsize=6.5, color="#555555")
    logo = logo_image()
    if logo is not None:
        logo_ax = fig.add_axes([.85, .002, .14, .056])
        logo_ax.imshow(logo)
        logo_ax.set_axis_off()
    else:
        fig.text(.986, .018, "SIDERAL", ha="right", va="bottom", fontsize=10, fontweight="bold", color="#c21f30")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", dpi=130)
    plt.close(fig)
    if not output.exists() or output.stat().st_size < 1000:
        raise RuntimeError(f"PNG inválido: {output}")
    return output


def render_all_views(lon, lat, values, key, title, valid, output, source,
                      region_root, region_relative, unit="", wind_u=None, wind_v=None,
                      wind_direction=None, unstructured=False, run_time=None,
                      forecast_hour=None, model_label=None):
    """Write one continental bulletin and five regional products from the same native-grid field."""
    output = Path(output)
    region_root = Path(region_root)
    rel = Path(region_relative)
    views = [(None, EXTENT_SOUTH_AMERICA, output)]
    for region_key, spec in REGIONS.items():
        views.append((spec["label"], spec["extent"], region_root / region_key / rel))
    written = []
    for region_name, view_extent, path in views:
        render_map(
            lon, lat, values, key, title, valid, path, source,
            unit=unit, wind_u=wind_u, wind_v=wind_v, wind_direction=wind_direction,
            unstructured=unstructured, extent=view_extent, region_name=region_name,
            run_time=run_time, forecast_hour=forecast_hour, model_label=model_label,
        )
        written.append(path)
    return written
