import argparse, json, tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import maximum_filter, minimum_filter
from ecmwf.opendata import Client

EXTENT = (-90, -30, -60, 15)
STEPS = list(range(0, 145, 3)) + list(range(150, 241, 6))
PRECIP_COLORS = ['#eef8ee', '#c7e9c0', '#74c476', '#31a354', '#006d2c', '#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
PRECIP_LEVELS = [0.05, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 80]
PRECIP_CMAP = LinearSegmentedColormap.from_list('sideral_precip', PRECIP_COLORS, N=len(PRECIP_COLORS))
PRECIP_NORM = BoundaryNorm(PRECIP_LEVELS, PRECIP_CMAP.N, clip=True)


def latest_available_run():
    now = datetime.now(timezone.utc)
    candidate = now - timedelta(hours=8)
    hour = 12 if candidate.hour >= 12 else 0
    return candidate.replace(hour=hour, minute=0, second=0, microsecond=0)


def retrieve(client, run, target_dir):
    date, hour = run.strftime('%Y%m%d'), run.hour
    surface = target_dir / 'surface.grib2'
    msl = target_dir / 'msl.grib2'
    pressure = target_dir / 'pressure.grib2'

    if not surface.exists():
        client.retrieve(date=date, time=hour, stream='oper', type='fc', step=STEPS,
                       levtype='sfc', param=['tp', '2t'], target=str(surface))

    if not msl.exists():
        client.retrieve(date=date, time=hour, stream='oper', type='fc', step=STEPS,
                       levtype='msl', param='msl', target=str(msl))

    if not pressure.exists():
        client.retrieve(date=date, time=hour, stream='oper', type='fc', step=STEPS,
                       levtype='pl', levelist=[1000, 500], param='gh', target=str(pressure))

    return surface, msl, pressure


def open_field(path, level=None, mean_sea=False):
    if mean_sea:
        keys = {'typeOfLevel': 'meanSea'}
    elif level is None:
        keys = {'typeOfLevel': 'surface'}
    else:
        keys = {'typeOfLevel': 'isobaricInhPa', 'level': level}
    return xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': '', 'filter_by_keys': keys})


def field(ds, name, step):
    da = ds[name]
    if 'step' in da.dims:
        da = da.sel(step=np.timedelta64(step, 'h'))
    return da.values


def add_extrema(ax, lon, lat, mslp):
    if np.asarray(mslp).ndim != 2:
        return
    for mask, label in ((maximum_filter(mslp, size=21) == mslp, 'H'),
                        (minimum_filter(mslp, size=21) == mslp, 'L')):
        iy, ix = np.where(mask)
        candidates = sorted(zip(iy, ix), key=lambda p: abs(mslp[p] - np.nanmedian(mslp[max(0,p[0]-2):p[0]+3, max(0,p[1]-2):p[1]+3])))[:10]
        used = []
        for y, x in candidates:
            xlon, ylat = float(lon[x]), float(lat[y])
            if not (EXTENT[0] <= xlon <= EXTENT[1] and EXTENT[2] <= ylat <= EXTENT[3]):
                continue
            if any((xlon-u[0])**2 + (ylat-u[1])**2 < 100 for u in used):
                continue
            used.append((xlon, ylat))
            color = '#1683d8' if label == 'H' else '#e22b2b'
            ax.text(xlon, ylat, label, transform=ccrs.PlateCarree(), ha='center', va='center',
                    fontsize=15, fontweight='bold', color=color,
                    bbox=dict(boxstyle='square,pad=0.08', facecolor='white', edgecolor=color, alpha=.78), zorder=8)


def make_map(surface_path, msl_path, pressure_path, run, step, out):
    sfc = open_field(surface_path)
    msl_ds = open_field(msl_path, mean_sea=True)
    gh1000, gh500 = open_field(pressure_path, 1000), open_field(pressure_path, 500)

    lon, lat = sfc['longitude'].values, sfc['latitude'].values
    tp = field(sfc, 'tp', step) * 1000.0
    if step == 0:
        rate = np.zeros_like(tp)
    else:
        prev = step - (3 if step <= 144 else 6)
        rate = np.maximum((tp - field(sfc, 'tp', prev) * 1000.0) / (step - prev), 0)

    msl = field(msl_ds, 'msl', step) / 100.0
    thickness = field(gh500, 'gh', step) - field(gh1000, 'gh', step)

    fig = plt.figure(figsize=(14, 8.2), dpi=130)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT)
    ax.set_facecolor('#f7f7f7')
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#fafafa', edgecolor='#bcbcbc', linewidth=.4, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=.45, edgecolor='#777', zorder=3)
    ax.coastlines(resolution='110m', linewidth=.7, color='#333', zorder=4)

    cf = ax.contourf(lon, lat, rate, levels=PRECIP_LEVELS, cmap=PRECIP_CMAP, norm=PRECIP_NORM,
                     alpha=.62, extend='max', transform=ccrs.PlateCarree(), zorder=2)

    tc = ax.contour(lon, lat, thickness, levels=np.arange(528, 600, 6), colors='#e51c23',
                    linewidths=.8, linestyles='--', transform=ccrs.PlateCarree(), zorder=5)
    ax.clabel(tc, fmt='%d', fontsize=7, inline=True)

    pc = ax.contour(lon, lat, msl, levels=np.arange(960, 1041, 4), colors='#111',
                    linewidths=.65, transform=ccrs.PlateCarree(), zorder=6)
    ax.clabel(pc, fmt='%d', fontsize=7, inline=True)
    add_extrema(ax, lon, lat, msl)

    valid = run + timedelta(hours=step)
    fig.suptitle(f'Precipitation Type, Rate (mm hr⁻¹), 1000–500 mb Thickness\nF{step:03d} Valid: {valid:%a %Y-%m-%d %HZ}',
                 fontsize=15, fontweight='bold', x=.52, y=.975)
    ax.set_title(f'Init: {run:%a %Y-%m-%d %HZ} ECMWF IFS HRES  •  South America', fontsize=9, loc='left', pad=8)

    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=.035, fraction=.045, aspect=45)
    cbar.set_label('Precipitation rate (mm h⁻¹)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    gl = ax.gridlines(draw_labels=True, linewidth=.25, color='#999', alpha=.35, linestyle=':')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    fig.text(.995, .005, 'ECMWF open data • Sideral Meteorologia', ha='right', va='bottom', fontsize=7, color='#555')
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    sfc.close()
    msl_ds.close()
    gh1000.close()
    gh500.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='site')
    args = ap.parse_args()

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    run = latest_available_run()
    run_id = run.strftime('%Y%m%d%H')
    run_dir = root / 'maps' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    client = Client(source='ecmwf', model='ifs', resol='0p25', infer_stream_keyword=True)

    with tempfile.TemporaryDirectory() as tmp:
        surface, msl, pressure = retrieve(client, run, Path(tmp))
        for step in STEPS:
            out = run_dir / f'f{step:03d}.png'
            if not out.exists():
                make_map(surface, msl, pressure, run, step, out)

    manifest = {
        'model':'ECMWF IFS HRES',
        'resolution':'0.25°',
        'run':run.isoformat(),
        'run_id':run_id,
        'steps':STEPS,
        'updated':datetime.now(timezone.utc).isoformat()
    }

    (root/'maps'/'index.json').write_text(
        json.dumps(manifest, indent=2),
        encoding='utf-8'
    )

    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
