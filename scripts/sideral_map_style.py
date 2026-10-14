"""Shared Sideral map renderer, visually aligned with the ECMWF publication pipeline."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import maximum_filter, minimum_filter

EXTENT = (-90, -30, -60, 15)
PRECIP_COLORS = ["#eef8ee", "#c7e9c0", "#74c476", "#31a354", "#006d2c",
                 "#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
PRECIP_LEVELS = [0.05, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 80]


def base_axes():
    fig = plt.figure(figsize=(14, 8.2), dpi=130)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("#f7f7f7")
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#fafafa", edgecolor="#bcbcbc", linewidth=.4, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=.45, edgecolor="#777", zorder=3)
    ax.coastlines(resolution="110m", linewidth=.7, color="#333", zorder=4)
    return fig, ax


def finish(fig, ax, title, unit, mappable, source):
    fig.suptitle(title, fontsize=15, fontweight="bold", x=.52, y=.975)
    gl = ax.gridlines(draw_labels=True, linewidth=.25, color="#999", alpha=.35, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=.035, fraction=.045, aspect=45)
        cbar.set_label(unit, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    fig.text(.995, .005, f"{source} • Sideral Meteorologia", ha="right", va="bottom", fontsize=7, color="#555")


def _add_extrema(ax, lon, lat, data):
    if data.ndim != 2 or lon.ndim != 1 or lat.ndim != 1 or data.shape != (lat.size, lon.size):
        return
    for mask, label in ((maximum_filter(data, size=21) == data, "H"),
                        (minimum_filter(data, size=21) == data, "L")):
        iy, ix = np.where(mask & np.isfinite(data))
        center = float(np.nanmedian(data))
        candidates = sorted(
            zip(iy, ix),
            key=lambda p: abs(data[p] - np.nanmedian(
                data[max(0, p[0]-2):p[0]+3, max(0, p[1]-2):p[1]+3]
            )) if np.isfinite(data[max(0,p[0]-2):p[0]+3,max(0,p[1]-2):p[1]+3]).any() else abs(data[p]-center)
        )[:10]
        used = []
        for y, x in candidates:
            xlon, ylat = float(lon[x]), float(lat[y])
            if not (EXTENT[0] <= xlon <= EXTENT[1] and EXTENT[2] <= ylat <= EXTENT[3]):
                continue
            if any((xlon-u[0])**2 + (ylat-u[1])**2 < 100 for u in used):
                continue
            used.append((xlon, ylat))
            color = "#1683d8" if label == "H" else "#e22b2b"
            ax.text(xlon, ylat, label, transform=ccrs.PlateCarree(), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color,
                    bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor=color, alpha=.78),
                    zorder=8)


def _family(key: str) -> str:
    k = key.lower()
    if "precip" in k or k in {"tp", "apcp"}:
        return "precip"
    if "cape" in k:
        return "cape"
    if "cloud" in k or "tcc" in k or "clct" in k:
        return "cloud"
    if "tcwv" in k or "water_vapour" in k or "pwat" in k or "tqv" in k:
        return "tcwv"
    if "pressure" in k or k in {"mslp", "prmsl"}:
        return "pressure"
    if "height" in k or k.startswith(("gh", "hgt")) or "geopotential" in k:
        return "height"
    if "wind" in k or k.startswith(("u", "v")) or "gust" in k:
        return "wind"
    if "temp" in k or "dew" in k or k in {"t2m", "2t", "2d", "t850"}:
        return "temperature"
    return "generic"


def render_map(lon, lat, values, key, title, valid, output, source,
               unit="", wind_u=None, wind_v=None, wind_direction=None,
               unstructured=False):
    """Render regular lon/lat grids and ICON-style scattered grids with ECMWF-like layout."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    data = np.asarray(values, dtype=float).squeeze().copy()
    family = _family(key)
    source_unit = str(unit or "").strip().lower()
    out_unit = unit or ""
    data[~np.isfinite(data)] = np.nan

    if family == "temperature":
        if np.isfinite(data).any() and float(np.nanmedian(data)) > 150:
            data -= 273.15
        out_unit = "°C"
    elif family == "pressure":
        if np.isfinite(data).any() and float(np.nanmedian(data)) > 2000:
            data /= 100.0
        out_unit = "hPa"
    elif family == "height":
        if np.isfinite(data).any() and float(np.nanmedian(np.abs(data))) > 50000:
            data /= 9.80665
        out_unit = "m"
    elif family == "cloud":
        if np.isfinite(data).any() and float(np.nanpercentile(data, 95)) <= 1.01:
            data *= 100.0
        data = np.clip(data, 0, 100)
        out_unit = "%"
    elif family == "precip":
        data = np.maximum(data, 0)
        out_unit = "mm"
    elif family == "tcwv":
        data = np.maximum(data, 0)
        out_unit = "kg m⁻²"
    elif family == "cape":
        data = np.maximum(data, 0)
        out_unit = "J kg⁻¹"

    if family == "wind":
        factor = 1.0
        if "kn" in source_unit or "knot" in source_unit:
            factor = 1.852
        elif source_unit in ("m/s", "m s-1", "m s⁻¹", "ms-1") or "m s" in source_unit:
            factor = 3.6
        data *= factor
        out_unit = "km h⁻¹"
        if wind_u is not None:
            wind_u = np.asarray(wind_u, dtype=float)
        if wind_v is not None:
            wind_v = np.asarray(wind_v, dtype=float)
        if wind_direction is not None and wind_u is None and wind_v is None:
            raw_speed = np.asarray(values, dtype=float).squeeze()
            direction = np.deg2rad(np.asarray(wind_direction, dtype=float).squeeze())
            wind_u = -raw_speed * np.sin(direction)
            wind_v = -raw_speed * np.cos(direction)

    fig, ax = base_axes()
    is_scattered = unstructured or (
        data.ndim == 1 and lon.ndim == 1 and lat.ndim == 1
        and lon.size == data.size and lat.size == data.size
    )
    if is_scattered:
        if not (lon.ndim == lat.ndim == data.ndim == 1 and lon.size == lat.size == data.size):
            raise ValueError(f"Grade dispersa incompatível: lon={lon.shape}, lat={lat.shape}, values={data.shape}")
        valid_mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(data)
        lonp, latp, datap = lon[valid_mask], lat[valid_mask], data[valid_mask]
        if datap.size < 20:
            raise ValueError(f"Poucos pontos válidos para {key}: {datap.size}")
        tri = mtri.Triangulation(lonp, latp)
        triangles = tri.triangles
        x, y = lonp[triangles], latp[triangles]
        max_edge = np.maximum.reduce([
            np.hypot(x[:,0]-x[:,1], y[:,0]-y[:,1]),
            np.hypot(x[:,1]-x[:,2], y[:,1]-y[:,2]),
            np.hypot(x[:,2]-x[:,0], y[:,2]-y[:,0]),
        ])
        tri.set_mask((max_edge > 2.0) | np.any(~np.isfinite(datap[triangles]), axis=1))
        plot_lon, plot_lat, plot_data = lonp, latp, datap
        def draw(levels, cmap, **kw):
            return ax.tricontourf(tri, datap, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2, **kw)
    else:
        if lon.ndim == 1 and lat.ndim == 1:
            if data.ndim != 2 or data.shape != (lat.size, lon.size):
                raise ValueError(f"Grade incompatível: values={data.shape}; lat={lat.shape}; lon={lon.shape}")
            plot_lon, plot_lat, plot_data = lon, lat, data
            def draw(levels, cmap, **kw):
                return ax.contourf(lon, lat, data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2, **kw)
        elif lon.ndim == 2 and lat.ndim == 2 and data.shape == lon.shape == lat.shape:
            plot_lon, plot_lat, plot_data = lon, lat, data
            def draw(levels, cmap, **kw):
                return ax.contourf(lon, lat, data, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), zorder=2, **kw)
        else:
            raise ValueError(f"Grade não suportada: values={data.shape}, lat={lat.shape}, lon={lon.shape}")

    if family == "precip":
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("sideral_precip", PRECIP_COLORS, N=len(PRECIP_COLORS))
        mappable = draw(PRECIP_LEVELS, cmap, extend="max", alpha=.72)
    elif family == "temperature":
        k = key.lower()
        if "250" in k:
            levels = np.arange(-80, -19, 2)
        elif "500" in k:
            levels = np.arange(-70, 1, 2)
        elif "850" in k:
            levels = np.arange(-30, 31, 2)
        else:
            levels = np.arange(-10, 41, 2)
        mappable = draw(levels, "RdYlBu_r", extend="both")
    elif family == "wind":
        upper = 120 if any(s in key.lower() for s in ("850", "500", "250")) else 80
        mappable = draw(np.arange(0, upper + 1, 5), "viridis", extend="max")
        if wind_u is not None and wind_v is not None:
            try:
                if is_scattered:
                    mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(np.asarray(values).squeeze())
                    ui, vi = wind_u.squeeze()[mask], wind_v.squeeze()[mask]
                    stride = max(1, plot_lon.size // 700)
                    ax.barbs(plot_lon[::stride], plot_lat[::stride], ui[::stride], vi[::stride],
                             length=4.5, linewidth=.35, transform=ccrs.PlateCarree(), zorder=5)
                elif lon.ndim == 1 and lat.ndim == 1:
                    stride_x = max(1, len(lon)//35)
                    stride_y = max(1, len(lat)//25)
                    ax.barbs(lon[::stride_x], lat[::stride_y], wind_u[::stride_y, ::stride_x], wind_v[::stride_y, ::stride_x],
                             length=5, linewidth=.35, transform=ccrs.PlateCarree(), zorder=5)
                else:
                    stride = max(1, data.shape[1]//35, data.shape[0]//25)
                    ax.barbs(lon[::stride, ::stride], lat[::stride, ::stride],
                             wind_u[::stride, ::stride], wind_v[::stride, ::stride],
                             length=5, linewidth=.35, transform=ccrs.PlateCarree(), zorder=5)
            except Exception as exc:
                print(f"[renderer] Aviso: barbelas não desenhadas para {key}: {exc}", flush=True)
    elif family == "pressure":
        levels = np.arange(960, 1041, 2)
        mappable = draw(levels, "viridis", extend="both", alpha=.25)
        try:
            if not is_scattered and lon.ndim == lat.ndim == 1 and plot_data.shape == (lat.size, lon.size):
                contours = ax.contour(lon, lat, plot_data, levels=np.arange(960, 1041, 4),
                                      colors="#111", linewidths=.65, transform=ccrs.PlateCarree(), zorder=6)
                ax.clabel(contours, fmt="%d", fontsize=7, inline=True)
                _add_extrema(ax, lon, lat, plot_data)
        except Exception:
            pass
    elif family == "cloud":
        mappable = draw(np.arange(0, 101, 10), "Greys", extend="neither")
    elif family == "tcwv":
        mappable = draw(np.arange(0, 71, 5), "GnBu", extend="max")
    elif family == "cape":
        mappable = draw([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000], "magma", extend="max")
    elif family == "height":
        k = key.lower()
        finite = plot_data[np.isfinite(plot_data)]
        if "250" in k:
            lo, hi, gap = 9000, 12000, 100
        elif "850" in k:
            lo, hi, gap = 1200, 1800, 30
        else:
            lo, hi, gap = 4800, 6000, 60
        if not finite.size or np.nanmedian(finite) < lo - gap*3 or np.nanmedian(finite) > hi + gap*3:
            lo, hi = (np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)) if finite.size else (0, 1)
            gap = max(1, (hi-lo)/20)
        levels = np.arange(np.floor(lo/gap)*gap, np.ceil(hi/gap)*gap + 1, gap) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else 20
        mappable = draw(levels, "viridis", extend="both")
        if not is_scattered and lon.ndim == lat.ndim == 1 and plot_data.shape == (lat.size, lon.size):
            try:
                contours = ax.contour(lon, lat, plot_data, levels=levels, colors="#222", linewidths=.45,
                                      transform=ccrs.PlateCarree(), zorder=5)
                ax.clabel(contours, fmt="%d", fontsize=6)
            except Exception:
                pass
    else:
        finite = plot_data[np.isfinite(plot_data)]
        levels = np.linspace(np.nanpercentile(finite, 2), np.nanpercentile(finite, 98), 25) if finite.size > 1 else 20
        mappable = draw(levels, "viridis", extend="both")

    finish(fig, ax, title, out_unit, mappable, source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not output.exists() or output.stat().st_size < 1000:
        raise RuntimeError(f"PNG inválido: {output}")
