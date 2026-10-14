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
from eccodes import codes_get, codes_grib_new_from_file, codes_release

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
    'cape': {'label': 'MUCAPE (parcela mais instável)', 'file': 'cape', 'group': 'surface'},
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
                       param=['tp', '2t', '2d', '10u', '10v', '10fg', '10fg3', 'msl', 'tcc', 'tcwv', 'mucape'],
                       target=str(surface))
    if not pressure.exists():
        client.retrieve(date=date, time=hour, stream='oper', type='fc', step=STEPS,
                       levtype='pl', levelist=[850, 500, 250],
                       param=['gh', 't', 'u', 'v'], target=str(pressure))
    return surface, pressure


# Reuse the successful GRIB selector for each field across forecast hours.
# ECMWF's API "levtype=sfc" is not the same thing as ecCodes' typeOfLevel.
_OPEN_FIELD_STRATEGIES = {}
_FIELD_ALIASES = {
    '2t': ('2t', 't2m'), '2d': ('2d', 'd2m'),
    '10u': ('10u', 'u10'), '10v': ('10v', 'v10'),
    '10fg': ('10fg', '10fg3', 'fg10', 'fg310', 'fg10_3'),
    '10fg3': ('10fg3', '10fg', 'fg10', 'fg310', 'fg10_3'),
    'tcc': ('tcc',), 'tp': ('tp',), 'tcwv': ('tcwv',),
    'mucape': ('mucape', 'cape'), 'cape': ('cape', 'mucape'),
    'msl': ('msl',), 't': ('t',), 'u': ('u',), 'v': ('v',),
    'gh': ('gh',),
}


def _dataset_has_requested_field(ds, short_name):
    if short_name is None:
        return bool(ds.data_vars)
    accepted = set(_FIELD_ALIASES.get(short_name, (short_name,)))
    for name, da in ds.data_vars.items():
        if name in accepted or da.attrs.get('GRIB_shortName') in accepted:
            return True
    return False


def open_field(path, level=None, type_level=None, short_name=None):
    # Prefer the precise level selector, then relax only typeOfLevel while
    # retaining shortName and (when supplied) level. ECMWF's API levtype=sfc
    # does not equal a single ecCodes/cfgrib typeOfLevel for every parameter.
    if type_level is not None:
        exact = {'typeOfLevel': type_level}
        if level is not None:
            exact['level'] = level
    elif level is None:
        exact = {'typeOfLevel': 'surface'}
    else:
        exact = {'typeOfLevel': 'isobaricInhPa', 'level': level}
    if short_name is not None:
        exact['shortName'] = short_name

    selectors = [exact]
    if short_name is not None and level is not None:
        # Never drop an explicitly requested level; that could return 500 hPa
        # when the caller asked for 850 hPa.
        selectors.append({'shortName': short_name, 'level': level})
    elif short_name is not None:
        selectors.append({'shortName': short_name})
    unique_selectors = []
    for keys in selectors:
        if keys not in unique_selectors:
            unique_selectors.append(keys)

    cache_key = (str(Path(path).resolve()), level, type_level, short_name)
    cached = _OPEN_FIELD_STRATEGIES.get(cache_key)
    if cached is not None:
        unique_selectors = [cached] + [keys for keys in unique_selectors if keys != cached]
    errors = []
    for keys in unique_selectors:
        ds = None
        try:
            ds = xr.open_dataset(
                path, engine='cfgrib', decode_timedelta=False,
                backend_kwargs={'indexpath': '', 'filter_by_keys': keys},
            )
            if _dataset_has_requested_field(ds, short_name):
                _OPEN_FIELD_STRATEGIES[cache_key] = keys
                return ds
            errors.append(f"{keys}: no matching data variable (found {list(ds.data_vars)})")
            ds.close()
        except Exception as exc:
            # cfgrib/eccodes exceptions vary by the precise filter and version;
            # record the rejection and try the next safe selector.
            errors.append(f"{keys}: {type(exc).__name__}: {exc}")
            if ds is not None:
                ds.close()

    raise KeyError(
        f"ECMWF GRIB field unavailable: shortName={short_name!r}, "
        f"typeOfLevel={type_level!r}, level={level!r}; " + " | ".join(errors)
    )


def _safe_codes_get(gid, key, default=None):
    try:
        return codes_get(gid, key)
    except Exception:
        return default


def inventory_grib(path):
    """Inspect GRIB message headers only; do not decompress grid values."""
    grouped = {}
    message_count = 0
    read_error = None
    try:
        with open(path, 'rb') as handle:
            while True:
                gid = codes_grib_new_from_file(handle, headers_only=True)
                if gid is None:
                    break
                message_count += 1
                try:
                    short_name = str(_safe_codes_get(gid, 'shortName', 'unknown'))
                    param_id = _safe_codes_get(gid, 'paramId')
                    name = str(_safe_codes_get(gid, 'name', ''))
                    type_of_level = str(_safe_codes_get(gid, 'typeOfLevel', 'unknown'))
                    level = _safe_codes_get(gid, 'level')
                    step_type = str(_safe_codes_get(gid, 'stepType', 'unknown'))
                    start_step = _safe_codes_get(gid, 'startStep')
                    end_step = _safe_codes_get(gid, 'endStep')
                    step_range = str(_safe_codes_get(gid, 'stepRange', ''))
                    key = (short_name, str(param_id), type_of_level, str(level), step_type)
                    item = grouped.setdefault(key, {
                        'shortName': short_name, 'paramId': param_id, 'name': name,
                        'typeOfLevel': type_of_level, 'level': level,
                        'stepType': step_type, 'steps': set(), 'stepRanges': set(),
                        'messages': 0,
                    })
                    item['messages'] += 1
                    try:
                        item['steps'].add(int(float(end_step)))
                    except (TypeError, ValueError):
                        pass
                    if step_range:
                        item['stepRanges'].add(step_range)
                    if start_step is not None:
                        item.setdefault('startSteps', set()).add(str(start_step))
                finally:
                    codes_release(gid)
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {exc}"

    fields = []
    for item in grouped.values():
        item['steps'] = sorted(item['steps'])
        item['stepRanges'] = sorted(item['stepRanges'])
        item['startSteps'] = sorted(item.get('startSteps', set()))
        fields.append(item)
    fields.sort(key=lambda x: (
        x['shortName'], x['typeOfLevel'], str(x['level']),
        x['stepType'], str(x['paramId'])
    ))
    report = {
        'file': Path(path).name,
        'sizeBytes': Path(path).stat().st_size if Path(path).exists() else None,
        'messageCount': message_count,
        'fields': fields,
    }
    if read_error:
        report['readError'] = read_error
    return report


FIELD_REQUIREMENTS = {
    'precip': [('surface', ('tp',), None)],
    'temp2m': [('surface', ('2t', 't2m'), 2)],
    'dew2m': [('surface', ('2d', 'd2m'), 2)],
    'wind10m': [('surface', ('10u', 'u10'), 10), ('surface', ('10v', 'v10'), 10)],
    'gust10m': [('surface', ('10fg', '10fg3', 'fg10', 'fg310', 'fg10_3'), None)],
    'mslp': [('surface', ('msl',), None)],
    'cloud': [('surface', ('tcc',), None)],
    'tcwv': [('surface', ('tcwv',), None)],
    # Current ECMWF Open Data lists MUCAPE, not the generic CAPE parameter.
    'cape': [('surface', ('mucape', 'cape'), None)],
    't850': [('pressure', ('t',), 850)],
    'wind850': [('pressure', ('u',), 850), ('pressure', ('v',), 850)],
    'gh500': [('pressure', ('gh',), 500)],
    'wind500': [('pressure', ('u',), 500), ('pressure', ('v',), 500)],
    'wind250': [('pressure', ('u',), 250), ('pressure', ('v',), 250)],
}


def build_availability_report(inventory):
    """Compare requested products against actual parameter names, levels and steps."""
    files = {'surface': inventory.get('surface', {}), 'pressure': inventory.get('pressure', {})}
    result = {}
    for product, requirements in FIELD_REQUIREMENTS.items():
        requirement_steps = []
        requirement_details = []
        for file_key, names, required_level in requirements:
            file_report = files.get(file_key, {})
            fields = file_report.get('fields', [])
            matching = []
            steps = set()
            for item in fields:
                if item.get('shortName') not in names:
                    continue
                if required_level is not None:
                    try:
                        if not np.isclose(float(item.get('level')), required_level):
                            continue
                    except (TypeError, ValueError):
                        continue
                matching.append({
                    'shortName': item.get('shortName'),
                    'paramId': item.get('paramId'),
                    'typeOfLevel': item.get('typeOfLevel'),
                    'level': item.get('level'),
                    'stepType': item.get('stepType'),
                    'steps': item.get('steps', []),
                })
                steps.update(item.get('steps', []))
            requirement_steps.append(steps)
            requirement_details.append({
                'file': file_key,
                'acceptedShortNames': list(names),
                'requiredLevel': required_level,
                'matches': matching,
            })
        common_steps = set.intersection(*requirement_steps) if requirement_steps else set()
        result[product] = {
            'status': 'available' if common_steps else 'unavailable_or_unverified',
            'availableSteps': sorted(common_steps),
            'missingSteps': [step for step in STEPS if step not in common_steps],
            'requirements': requirement_details,
        }
    return result


def field(ds, name, step):
    # cfgrib/xarray exposes ECMWF GRIB short names with different aliases
    # depending on the selected typeOfLevel. Normalize the 2 m and 10 m
    # fields so the map generator does not depend on the backend naming.
    aliases = {
        '2t': ('2t', 't2m'),
        '2d': ('2d', 'd2m'),
        '10u': ('10u', 'u10'),
        '10v': ('10v', 'v10'),
        # ECMWF encodes gusts as 10fg at some steps and 10fg3
        # for the 3-hour post-processing intervals. Keep both aliases.
        # cfgrib may normalize the ECMWF 3-hour gust shortName 10fg3
        # to the variable name fg310. Include it for both gust aliases.
        '10fg': ('10fg', '10fg3', 'fg10', 'fg310', 'fg10_3'),
        '10fg3': ('10fg3', '10fg', 'fg10', 'fg310', 'fg10_3'),
        'tcc': ('tcc', 'tcc'),
        'tp': ('tp', 'tp'),
        'tcwv': ('tcwv', 'tcwv'),
        'cape': ('cape', 'cape'),
        'msl': ('msl', 'msl'),
        't': ('t', 't'),
        'u': ('u', 'u'),
        'v': ('v', 'v'),
        'gh': ('gh', 'gh'),
        'mucape': ('mucape', 'cape'),
    }
    candidates = aliases.get(name, (name,))
    found = [candidate for candidate in candidates if candidate in ds.variables]
    if not found:
        raise KeyError(f"No ECMWF field {name!r}; available variables: {list(ds.variables)}")

    # Resolve the forecast hour against each alias independently. Some GRIB
    # parameters are split between shortName aliases (notably 10fg/10fg3),
    # so the first alias may exist but not contain the requested step.
    step_errors = []
    for selected in found:
        da = ds[selected]
        if 'step' not in da.dims:
            if 'step' not in da.coords:
                return da.values
            raw_step = np.asarray(da.coords['step'].values).item()
            if isinstance(raw_step, np.timedelta64):
                available_hour = float(raw_step / np.timedelta64(1, 'h'))
            else:
                try:
                    available_hour = float(raw_step)
                except (TypeError, ValueError):
                    available_hour = float('nan')
            if np.isclose(available_hour, step, atol=1e-6):
                return da.values
            step_errors.append(f"{selected}: {available_hour:g} h")
            continue

        raw_steps = np.asarray(da.coords['step'].values).reshape(-1)
        if np.issubdtype(raw_steps.dtype, np.timedelta64):
            available_steps = raw_steps / np.timedelta64(1, 'h')
        else:
            try:
                available_steps = raw_steps.astype(float)
            except (TypeError, ValueError):
                available_steps = np.asarray([
                    float(value / np.timedelta64(1, 'h'))
                    if isinstance(value, np.timedelta64) else float(value)
                    for value in raw_steps
                ])
        matches = np.flatnonzero(np.isclose(available_steps, step, atol=1e-6))
        if matches.size:
            return da.isel(step=int(matches[0])).values
        step_errors.append(
            f"{selected}: {', '.join(f'{float(v):g}' for v in available_steps[:12])}"
            + (" ..." if len(available_steps) > 12 else "")
        )

    raise KeyError(
        f"No exact ECMWF forecast step {step} h for {name!r}; "
        f"available candidate steps: {'; '.join(step_errors)}"
    )


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
        ds_tp = open_field(surface_path, type_level='surface', short_name='tp')
        tp = field(ds_tp, 'tp', step) * 1000.0
        prev = step - (3 if step <= 144 else 6)
        rate = np.zeros_like(tp) if step == 0 else np.maximum((tp - field(ds_tp, 'tp', prev) * 1000.0) / (step - prev), 0)
        ds_tp.close()
        mappable = ax.contourf(lon, lat, rate, levels=PRECIP_LEVELS, cmap=PRECIP_CMAP, norm=PRECIP_NORM, alpha=.62, extend='max', transform=ccrs.PlateCarree(), zorder=2)
        unit = 'Precipitação (mm h⁻¹)'
        ax.set_title(f'Init: {run:%a %Y-%m-%d %HZ} ECMWF IFS HRES  •  South America', fontsize=9, loc='left', pad=8)
    elif product in ('temp2m', 'dew2m'):
        name = '2t' if product == 'temp2m' else '2d'
        ds2 = open_field(surface_path, level=2, type_level='heightAboveGround', short_name=name)
        val = field(ds2, name, step) - 273.15
        ds2.close()
        levels = np.arange(-10, 41, 2)
        mappable = ax.contourf(lon, lat, val, levels=levels, cmap='RdYlBu_r', extend='both', transform=ccrs.PlateCarree(), zorder=2)
        unit = '°C'
    elif product in ('wind10m', 'gust10m'):
        if product == 'wind10m':
            ds_u = open_field(surface_path, level=10, type_level='heightAboveGround', short_name='10u')
            ds_v = open_field(surface_path, level=10, type_level='heightAboveGround', short_name='10v')
            u, v = field(ds_u, '10u', step), field(ds_v, '10v', step)
            val = np.hypot(u, v) * 3.6
            ds_u.close(); ds_v.close()
        else:
            # ECMWF changes the gust parameter encoding across lead times.
            # Prefer the 10 m level, but also accept surface-level encodings:
            # 10fg is common at short leads, while 10fg3 is used for some
            # 3-hourly fields (notably F093-F144). Both are requested in retrieve().
            # Continue requiring an exact forecast hour; never substitute stale data.
            gust = None
            gust_errors = []
            for type_level, gust_level in (('heightAboveGround', 10), ('surface', None)):
                for short_name in ('10fg', '10fg3'):
                    dsfg = None
                    try:
                        dsfg = open_field(
                            surface_path, level=gust_level, type_level=type_level,
                            short_name=short_name,
                        )
                        gust = field(dsfg, '10fg', step)
                    except (KeyError, ValueError, OSError) as exc:
                        gust_errors.append(f"{type_level}/{short_name}: {exc}")
                        continue
                    finally:
                        if dsfg is not None:
                            dsfg.close()
                    if gust is not None:
                        break
                if gust is not None:
                    break
            if gust is None:
                raise RuntimeError(
                    f"Rajada ECMWF indisponível em F{step:03d}; "
                    + " | ".join(gust_errors)
                )
            val = gust * 3.6
        levels = np.arange(0, 81, 5)
        mappable = ax.contourf(lon, lat, val, levels=levels, cmap='viridis', extend='max', transform=ccrs.PlateCarree(), zorder=2)
        unit = 'km h⁻¹'
        if product == 'wind10m':
            sl = max(1, min(12, len(lon)//35)); ax.barbs(lon[::sl], lat[::sl], u[::sl, ::sl], v[::sl, ::sl], length=5, linewidth=.35, transform=ccrs.PlateCarree(), zorder=5)
    elif product == 'mslp':
        ds = open_field(surface_path, type_level='meanSea', short_name='msl')
        msl = field(ds, 'msl', step) / 100.0
        levels = np.arange(960, 1041, 2)
        mappable = ax.contourf(lon, lat, msl, levels=levels, cmap='viridis', alpha=.25, transform=ccrs.PlateCarree(), zorder=2)
        pc = ax.contour(lon, lat, msl, levels=np.arange(960, 1041, 4), colors='#111', linewidths=.65, transform=ccrs.PlateCarree(), zorder=6)
        ax.clabel(pc, fmt='%d', fontsize=7, inline=True); add_extrema(ax, lon, lat, msl); unit = 'hPa'
        ds.close()
    elif product == 'cloud':
        # ECMWF encodes total cloud cover as a surface-to-top-of-atmosphere
        # layer. Depending on the ecCodes/cfgrib version, its typeOfLevel can
        # be reported as entireAtmosphere/atmosphere rather than surface.
        # Try the recognized encodings and only accept the exact forecast step.
        val = None
        cloud_errors = []
        for type_level in ('entireAtmosphere', 'atmosphere', 'surface'):
            ds_tcc = None
            try:
                ds_tcc = open_field(
                    surface_path, type_level=type_level, short_name='tcc'
                )
                val = field(ds_tcc, 'tcc', step) * 100.0
                break
            except (KeyError, ValueError, OSError) as exc:
                cloud_errors.append(f"{type_level}: {exc}")
            finally:
                if ds_tcc is not None:
                    ds_tcc.close()
        if val is None:
            raise RuntimeError(
                f"Cobertura total de nuvens ECMWF indisponível em F{step:03d}; "
                + " | ".join(cloud_errors)
            )
        val = np.clip(val, 0.0, 100.0)
        mappable = ax.contourf(lon, lat, val, levels=np.arange(0, 101, 10), cmap='Greys', extend='neither', transform=ccrs.PlateCarree(), zorder=2)
        unit = '%'
    elif product == 'tcwv':
        ds_tcwv = open_field(surface_path, type_level='surface', short_name='tcwv')
        val = field(ds_tcwv, 'tcwv', step)
        ds_tcwv.close()
        mappable = ax.contourf(lon, lat, val, levels=np.arange(0, 71, 5), cmap='GnBu', extend='max', transform=ccrs.PlateCarree(), zorder=2)
        unit = 'kg m⁻²'
    elif product == 'cape':
        # The current IFS Open Data catalogue exposes most-unstable CAPE.
        # Generic CAPE (shortName=cape, paramId 59) is not in that feed list.
        ds_cape = open_field(surface_path, type_level='surface', short_name='mucape')
        val = np.maximum(field(ds_cape, 'mucape', step), 0)
        ds_cape.close()
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
        # Inspect GRIB headers first. This reports true availability separately
        # from cfgrib's decoded variable names and level filters.
        inventory = {
            'generatedAt': datetime.now(timezone.utc).isoformat(),
            'model': 'ECMWF IFS HRES',
            'run': run.isoformat(),
            'surface': inventory_grib(surface),
            'pressure': inventory_grib(pressure),
        }
        availability = build_availability_report(inventory)
        inventory['productAvailability'] = availability
        (run_dir / 'field-inventory.json').write_text(
            json.dumps(inventory, indent=2), encoding='utf-8'
        )

        failures = []
        generated_count = 0
        reused_count = 0
        inventory_ok = (
            not inventory['surface'].get('readError')
            and not inventory['pressure'].get('readError')
            and inventory['surface'].get('messageCount', 0) > 0
            and inventory['pressure'].get('messageCount', 0) > 0
        )
        for product in PRODUCTS:
            pdir = run_dir / product
            pdir.mkdir(parents=True, exist_ok=True)
            product_steps = set(availability.get(product, {}).get('availableSteps', []))
            for step in STEPS:
                out = pdir / f'f{step:03d}.png'
                if out.exists():
                    reused_count += 1
                    continue
                if inventory_ok and step not in product_steps:
                    failures.append({
                        'product': product, 'step': step,
                        'error': 'Parâmetro/horário ausente no inventário GRIB',
                    })
                    print(f"AVISO: {product} F{step:03d} ausente no inventário GRIB; seguindo para o próximo mapa.")
                    continue
                try:
                    make_map(surface, pressure, run, step, out, product)
                    generated_count += 1
                except Exception as exc:
                    failures.append({
                        'product': product, 'step': step,
                        'error': f'{type(exc).__name__}: {exc}',
                    })
                    print(f"AVISO: falha em {product} F{step:03d}: {type(exc).__name__}: {exc}")
                    plt.close('all')

        report = {
            'model': 'ECMWF IFS HRES',
            'resolution': '0.25°',
            'run': run.isoformat(),
            'run_id': run_id,
            'generatedAt': datetime.now(timezone.utc).isoformat(),
            'generatedMapsThisRun': generated_count,
            'existingMapsReused': reused_count,
            'failureCount': len(failures),
            'failures': failures,
            'inventoryFile': 'field-inventory.json',
            'productAvailability': availability,
        }
        (run_dir / 'generation-report.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8'
        )

    manifest = {
        'model': 'ECMWF IFS HRES', 'resolution': '0.25°',
        'run': run.isoformat(), 'run_id': run_id, 'steps': STEPS,
        'products': PRODUCTS, 'updated': datetime.now(timezone.utc).isoformat(),
        'fieldInventory': f'maps/{run_id}/field-inventory.json',
        'generationReport': f'maps/{run_id}/generation-report.json',
        'productAvailability': {
            product: {
                'status': details['status'],
                'availableStepCount': len(details['availableSteps']),
                'missingStepCount': len(details['missingSteps']),
            }
            for product, details in availability.items()
        },
    }
    (root/'maps'/'index.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
