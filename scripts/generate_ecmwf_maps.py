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

PRODUCTS = {
    'precip': {'label': 'Precipitação / taxa', 'file': 'precip', 'group': 'surface'},
    'temp2m': {'label': 'Temperatura 2 m', 'file': 'temp2m', 'group': 'surface'},
    'dew2m': {'label': 'Ponto de orvalho 2 m', 'file': 'dew2m', 'group': 'surface'},
    'wind10m': {'label': 'Vento 10 m', 'file': 'wind10m', 'group': 'surface'},
    'gust10m': {'label': 'Rajada máxima 10 m', 'file': 'gust10m', 'group': 'surface'},
    'mslp': {'label': 'Pressão ao nível do mar', 'file': 'mslp', 'group': 'surface'},
    'cloud': {'label': 'Cobertura de nuvens', 'file': 'cloud', 'group': 'surface'},
    'tcwv': {'label': 'Água precipitável (TCWV)', 'file': 'tcwv', 'group': 'surface'},
    'cape': {'label': 'CAPE', 'file': 'cape', 'group': 'surface'},
    't850': {'label': 'Temperatura 850 hPa', 'file': 't850', 'group': 'pressure'},
    'wind850': {'label': 'Vento 850 hPa', 'file': 'wind850', 'group': 'pressure'},
    'gh500': {'label': 'Geopotencial 500 hPa', 'file': 'gh500', 'group': 'pressure'},
    'wind500': {'label': 'Vento 500 hPa', 'file': 'wind500', 'group': 'pressure'},
    'wind250': {'label': 'Vento 250 hPa', 'file': 'wind250', 'group': 'pressure'},
}


def latest_available_run():
    now = datetime.now(timezone.utc)
    candidate = now - timedelta(hours=8)
    hour = 12 if candidate.hour >= 12 else 0
    return candidate.replace(hour=hour, minute=0, second=0, microsecond=0)


def retrieve(client, run, target_dir):
    date, hour = run.strftime('%Y%m%d'), run.hour
    surface = target_dir / 'surface.grib2'
    pressure = target_dir / 'pressure.grib2'
    if not surface.exists():
        client.retrieve(date=date, time=hour, stream='oper', type='fc', step=STEPS,
                       levtype='sfc',
                       param=['tp', '2t', '2d', '10u', '10v', '10fg', 'msl', 'tcc', 'tcwv', 'cape'],
                       target=str(surface))
    if not pressure.exists():
        client.retrieve(date=date, time=hour, stream='oper', type='fc', step=STEPS,
                       levtype='pl', levelist=[850, 500, 250],
                       param=['gh', 't', 'u', 'v'], target=str(pressure))
    return surface, pressure


def open_field(path, level=None, type_level=None):
    if type_level is not None:
        keys = {'typeOfLevel': type_level}
        if level is not None:
            keys['level'] = level
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


def base_axes():
    fig = plt.figure(figsize=(14, 8.2), dpi=130)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT)
    ax.set_facecolor('#f7f7f7')
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='#fafafa', edgecolor='#bcbcbc', linewidth=.4, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=.45, edgecolor='#777', zorder=3)
    ax.coastlines(resolution='110m', linewidth=.7, color='#333', zorder=4)
    return fig, ax


def finish(fig, ax, title, unit, mappable=None):
    fig.suptitle(title, fontsize=15, fontweight='bold', x=.52, y=.975)
    gl = ax.gridlines(draw_labels=True, linewidth=.25, color='#999', alpha=.35, linestyle=':')
    gl.top_labels = False; gl.right_labels = False
    gl.xlabel_style = {'size': 7}; gl.ylabel_style = {'size': 7}
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', pad=.035, fraction=.045, aspect=45)
        cbar.set_label(unit, fontsize=9); cbar.ax.tick_params(labelsize=8)
    fig.text(.995, .005, 'ECMWF open data • Sideral Meteorologia', ha='right', va='bottom', fontsize=7, color='#555')


def make_map(surface_path, pressure_path, run, step, out, product):
    sfc = open_field(surface_path)
    lon, lat = sfc['longitude'].values, sfc['latitude'].values
    valid = run + timedelta(hours=step)
    title = f"{PRODUCTS[product]['label']}\nF{step:03d} Valid: {valid:%a %Y-%m-%d %HZ}"
    fig, ax = base_axes()
    mappable = None
    unit = ''

    if product == 'precip':
        tp = field(sfc, 'tp', step) * 1000.0
        prev = step - (3 if step <= 144 else 6)
        rate = np.zeros_like(tp) if step == 0 else np.maximum((tp - field(sfc, 'tp', prev) * 1000.0) / (step - prev), 0)
        mappable = ax.contourf(lon, lat, rate, levels=PRECIP_LEVELS, cmap=PRECIP_CMAP, norm=PRECIP_NORM, alpha=.62, extend='max', transform=ccrs.PlateCarree(), zorder=2)
        unit = 'Precipitação (mm h⁻¹)'
        ax.set_title(f'Init: {run:%a %Y-%m-%d %HZ} ECMWF IFS HRES  •  South America', fontsize=9, loc='left', pad=8)
    elif product in ('temp2m', 'dew2m'):
        name = '2t' if product == 'temp2m' else '2d'
        val = field(open_field(surface_path, level=2, type_level='heightAboveGround'), name, step) - 273.15
        levels = np.arange(-10, 41, 2)
        mappable = ax.contourf(lon, lat, val, levels=levels, cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree(), zorder=2)
        unit = '°C'
    elif product in ('wind10m', 'gust10m'):
        ds10 = open_field(surface_path, level=10, type_level='heightAboveGround')
        if product == 'wind10m':
            u, v = field(ds10, '10u', step), field(ds10, '10v', step)
            val = np.hypot(u, v) * 3.6
        else:
            val = field(open_field(surface_path, type_level='heightAboveGround'), '10fg', step) * 3.6
        levels = np.arange(0, 81, 5)
        mappable = ax.contourf(lon, lat, val, levels=levels, cmap='viridis', extend='max', transform=ccrs.PlateCarree(), zorder=2)
        unit = 'km h⁻¹'
        if product == 'wind10m':
            sl = max(1, min(12, len(lon)//35)); ax.barbs(lon[::sl,] if lon.ndim else lon, lat[::sl,] if lat.ndim else lat, u[::sl, ::sl], v[::sl, ::sl], length=5, linewidth=.35, transform=ccrs.PlateCarree(), zorder=5)
    elif product == 'mslp':
        ds = open_field(surface_path, type_level='meanSea')
        msl = field(ds, 'msl', step) / 100.0
        levels = np.arange(960, 1041, 2)
        mappable = ax.contourf(lon, lat, msl, levels=levels, cmap='viridis', alpha=.25, transform=ccrs.PlateCarree(), zorder=2)
        pc = ax.contour(lon, lat, msl, levels=np.arange(960, 1041, 4), colors='#111', linewidths=.65, transform=ccrs.PlateCarree(), zorder=6)
        ax.clabel(pc, fmt='%d', fontsize=7, inline=True); add_extrema(ax, lon, lat, msl); unit = 'hPa'
        ds.close()
    elif product == 'cloud':
        val = field(sfc, 'tcc', step) * 100.0
        mappable = ax.contourf(lon, lat, val, levels=np.arange(0, 101, 10), cmap='Greys', extend='neither', transform=ccrs.PlateCarree(), zorder=2)
        unit = '%'
    elif product == 'tcwv':
        val = field(sfc, 'tcwv', step)
        mappable = ax.contourf(lon, lat, val, levels=np.arange(0, 71, 5), cmap='GnBu', extend='max', transform=ccrs.PlateCarree(), zorder=2)
        unit = 'kg m⁻²'
    elif product == 'cape':
        val = np.maximum(field(sfc, 'cape', step), 0)
        mappable = ax.contourf(lon, lat, val, levels=[0,250,500,1000,1500,2000,3000,4000,6000], cmap='magma', extend='max', transform=ccrs.PlateCarree(), zorder=2)
        unit = 'J kg⁻¹'
    else:
        level = {'t850':850, 'wind850':850, 'gh500':500, 'wind500':500, 'wind250':250}[product]
        ds = open_field(pressure_path, level=level)
        if product.startswith('t'):
            val = field(ds, 't', step) - 273.15
            mappable = ax.contourf(lon, lat, val, levels=np.arange(-30, 31, 2), cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree(), zorder=2); unit='°C'
        elif product == 'gh500':
            val = field(ds, 'gh', step)
            mappable = ax.contourf(lon, lat, val, levels=np.arange(4800, 6001, 60), cmap='viridis', transform=ccrs.PlateCarree(), zorder=2); unit='m'
            pc=ax.contour(lon,lat,val,levels=np.arange(4800,6001,60),colors='#222',linewidths=.45,transform=ccrs.PlateCarree(),zorder=5); ax.clabel(pc,fmt='%d',fontsize=6)
        else:
            u,v=field(ds,'u',step),field(ds,'v',step); val=np.hypot(u,v)*3.6
            mappable=ax.contourf(lon,lat,val,levels=np.arange(0,121,10),cmap='viridis',extend='max',transform=ccrs.PlateCarree(),zorder=2); unit='km h⁻¹'
        ds.close()

    finish(fig, ax, title, unit, mappable)
    fig.savefig(out, bbox_inches='tight', facecolor='white'); plt.close(fig); sfc.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', default='site'); args = ap.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    run = latest_available_run(); run_id = run.strftime('%Y%m%d%H'); run_dir = root / 'maps' / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    client = Client(source='ecmwf', model='ifs', resol='0p25', infer_stream_keyword=True)
    with tempfile.TemporaryDirectory() as tmp:
        surface, pressure = retrieve(client, run, Path(tmp))
        for product in PRODUCTS:
            pdir = run_dir / product; pdir.mkdir(parents=True, exist_ok=True)
            for step in STEPS:
                out = pdir / f'f{step:03d}.png'
                if not out.exists(): make_map(surface, pressure, run, step, out, product)
    manifest = {'model':'ECMWF IFS HRES','resolution':'0.25°','run':run.isoformat(),'run_id':run_id,'steps':STEPS,'products':PRODUCTS,'updated':datetime.now(timezone.utc).isoformat()}
    (root/'maps'/'index.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
