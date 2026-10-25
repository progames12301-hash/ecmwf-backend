#!/usr/bin/env python3
"""Operational DWD ICON-GLOBAL maps for South America; downloads only selected fields/steps."""
from __future__ import annotations
import bz2, json, re, shutil, urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sideral_map_style import render_all_views, REGIONS

EXTENT=(-90,-20,-60,15)
def steps_for_run(run):
 # DWD ICON Global: F000-F078 horário; depois a cada 3 h.
 # As rodadas 00/12Z chegam a F180; 06/18Z chegam a F120.
 end = 181 if run.hour in (0, 12) else 121
 return list(range(0, 79)) + list(range(81, end, 3))
BASE="https://opendata.dwd.de/weather/nwp/icon/grib"
CACHE=Path("work/icon-grib")
GRID_CACHE=Path("work/icon-grid")
COORDS_FILE=GRID_CACHE/"icon_coords_subset.npz"
GRID_URL="https://opendata.dwd.de/weather/lib/cdo/icon_grid_0026_R03B07_G.nc.bz2"
OUT=Path("site/maps/icon/global")
# DWD ICON-GLOBAL operational single-level fields. EU-only ICON is intentionally not used:
# it does not cover South America.
FIELDS={
 "temp2m":{"folder":"t_2m","suffix":"T_2M","label":"Temperatura a 2 m","unit":"°C","cmap":"coolwarm","vmin":-10,"vmax":40,"kind":"temp"},
 "dew2m":{"folder":"td_2m","suffix":"TD_2M","label":"Ponto de orvalho a 2 m","unit":"°C","cmap":"BrBG","vmin":-20,"vmax":30,"kind":"temp"},
 "precip":{"folder":"tot_prec","suffix":"TOT_PREC","label":"Precipitação acumulada","unit":"mm","cmap":"turbo","vmin":0,"vmax":80,"kind":"precip"},
 "cape":{"folder":"cape_con","suffix":"CAPE_CON","label":"CAPE convectiva","unit":"J/kg","cmap":"magma","vmin":0,"vmax":4000,"kind":"cape"},
 "mslp":{"folder":"pmsl","suffix":"PMSL","label":"Pressão ao nível do mar","unit":"hPa","cmap":"viridis","vmin":960,"vmax":1040,"kind":"pressure"},
 "cloud":{"folder":"clct","suffix":"CLCT","label":"Cobertura total de nuvens","unit":"%","cmap":"Greys","vmin":0,"vmax":100,"kind":"cloud"},
 "tcwv":{"folder":"tqv","suffix":"TQV","label":"Água precipitável total","unit":"kg m⁻²","cmap":"GnBu","vmin":0,"vmax":70,"kind":"tcwv"},
 "gust10m":{"folder":"vmax_10m","suffix":"VMAX_10M","label":"Rajada máxima a 10 m","unit":"m/s","cmap":"turbo","vmin":0,"vmax":40,"kind":"wind"},
}
WIND={"u":{"folder":"u_10m","suffix":"U_10M"},"v":{"folder":"v_10m","suffix":"V_10M"}}
PRESSURE_FIELDS={
 "t850":{"folder":"t","suffix":"T","level":850,"label":"Temperatura em 850 hPa","unit":"K"},
 "gh500":{"folder":"fi","suffix":"FI","level":500,"label":"Altura geopotencial em 500 hPa","unit":"m²/s²"},
 "wind850":{"level":850,"label":"Vento em 850 hPa","unit":"m/s","u_folder":"u","u_suffix":"U","v_folder":"v","v_suffix":"V"},
 "wind500":{"level":500,"label":"Vento em 500 hPa","unit":"m/s","u_folder":"u","u_suffix":"U","v_folder":"v","v_suffix":"V"},
 "wind250":{"level":250,"label":"Vento em 250 hPa","unit":"m/s","u_folder":"u","u_suffix":"U","v_folder":"v","v_suffix":"V"},
 "t700":{"folder":"t","suffix":"T","level":700,"label":"Temperatura em 700 hPa","unit":"K"},
 "t500":{"folder":"t","suffix":"T","level":500,"label":"Temperatura em 500 hPa","unit":"K"},
 "wind700":{"level":700,"label":"Vento em 700 hPa","unit":"m/s","u_folder":"u","u_suffix":"U","v_folder":"v","v_suffix":"V"},
 "wind300":{"level":300,"label":"Vento em 300 hPa","unit":"m/s","u_folder":"u","u_suffix":"U","v_folder":"v","v_suffix":"V"},
 "gh700":{"folder":"fi","suffix":"FI","level":700,"label":"Altura geopotencial em 700 hPa","unit":"m²/s²"},
 "gh300":{"folder":"fi","suffix":"FI","level":300,"label":"Altura geopotencial em 300 hPa","unit":"m²/s²"},
}


def latest_run():
 t=datetime.now(timezone.utc)-timedelta(hours=6)
 return t.replace(hour=(t.hour//6)*6,minute=0,second=0,microsecond=0)

_INDEX_CACHE={}
def links_from_index(url):
 if url in _INDEX_CACHE: return _INDEX_CACHE[url]
 req=urllib.request.Request(url,headers={"User-Agent":"SideralMeteorologia-map-runner/1.0"})
 with urllib.request.urlopen(req,timeout=60) as r: html=r.read().decode("utf-8","replace")
 names=re.findall(r"""href=["']([^"']+\.grib2\.bz2)["']""", html, re.I)
 _INDEX_CACHE[url]=names
 return names

def fetch_grib(run,step,folder,suffix,level=None):
 cycle=f"{run.hour:02d}"
 directory=f"{BASE}/{cycle}/{folder}/"
 if level is None:
  filename=f"icon_global_icosahedral_single-level_{run:%Y%m%d%H}_{step:03d}_{suffix}.grib2.bz2"
  local_name=f"{suffix}.grib2"
 else:
  filename=f"icon_global_icosahedral_pressure-level_{run:%Y%m%d%H}_{step:03d}_{level}_{suffix}.grib2.bz2"
  local_name=f"{suffix}_{level}.grib2"
 local_dir=CACHE/f"{run:%Y%m%d%H}"/f"{step:03d}"
 local_dir.mkdir(parents=True,exist_ok=True)
 out=local_dir/local_name
 if out.exists() and out.stat().st_size>1000: return out
 names=links_from_index(directory)
 target=next((n for n in names if n.endswith(filename)),None)
 if not target:
  # Exact expected naming was not present: report the live folder entries for diagnostics.
  candidates=[n for n in names if run.strftime("%Y%m%d%H") in n and f"_{step:03d}_" in n]
  raise FileNotFoundError(f"ICON não publicou {filename} em {directory}; candidatos para a hora: {candidates[:8]}")
 url=urljoin(directory,target)
 req=urllib.request.Request(url,headers={"User-Agent":"SideralMeteorologia-map-runner/1.0"})
 with urllib.request.urlopen(req,timeout=180) as r: payload=r.read()
 # DWD files ending in .bz2 may be either genuinely bz2-compressed or raw GRIB2.
 if payload[:3]==b"BZh": payload=bz2.decompress(payload)
 if not payload.startswith(b"GRIB"): raise ValueError(f"Resposta não parece GRIB2: {url}")
 out.write_bytes(payload)
 return out

def ensure_coordinates():
 # ICON global GRIB stores values on its native unstructured grid without lon/lat.
 # Cache the relevant grid-point coordinates extracted from DWD's official grid file.
 if COORDS_FILE.exists() and COORDS_FILE.stat().st_size>1000:
  with np.load(COORDS_FILE) as saved:
   if all(k in saved for k in ("indices","lon","lat","total")):
    return
 GRID_CACHE.mkdir(parents=True,exist_ok=True)
 compressed=GRID_CACHE/"icon_grid_0026_R03B07_G.nc.bz2"
 grid_path=GRID_CACHE/"icon_grid_0026_R03B07_G.nc"
 if not compressed.exists() or compressed.stat().st_size<100000000:
  req=urllib.request.Request(GRID_URL,headers={"User-Agent":"SideralMeteorologia-map-runner/1.0"})
  temp=compressed.with_suffix(".download")
  with urllib.request.urlopen(req,timeout=900) as response, open(temp,"wb") as out:
   shutil.copyfileobj(response,out,length=1024*1024)
  if temp.stat().st_size<100000000:
   temp.unlink(missing_ok=True)
   raise ValueError("Download incompleto do grid ICON Global: arquivo menor que 100 MB")
  temp.replace(compressed)
 if not grid_path.exists() or grid_path.stat().st_size<1000000000:
  with bz2.open(compressed,"rb") as src, open(grid_path,"wb") as dst:
   shutil.copyfileobj(src,dst,length=1024*1024)
 if grid_path.stat().st_size<1000000000:
  raise ValueError("O grid ICON Global descompactado parece incompleto")
 with Dataset(str(grid_path),"r") as grid:
  if "clon" not in grid.variables or "clat" not in grid.variables:
   raise ValueError(f"Grid DWD sem clon/clat: {list(grid.variables)[:30]}")
  lon=np.rad2deg(np.asarray(grid.variables["clon"][:],dtype=np.float64))
  lat=np.rad2deg(np.asarray(grid.variables["clat"][:],dtype=np.float64))
 lon=((lon+180.0)%360.0)-180.0
 mask=(lon>=EXTENT[0]-5)&(lon<=EXTENT[1]+5)&(lat>=EXTENT[2]-5)&(lat<=EXTENT[3]+5)
 indices=np.flatnonzero(mask).astype(np.int32)
 if indices.size<1000:
  raise ValueError(f"Poucos pontos do grid ICON na área solicitada: {indices.size}")
 np.savez_compressed(COORDS_FILE,indices=indices,lon=lon[mask].astype(np.float32),
                     lat=lat[mask].astype(np.float32),total=np.int64(lon.size))
 # Only the compact coordinate subset is needed for subsequent runs.
 grid_path.unlink(missing_ok=True)
 compressed.unlink(missing_ok=True)
 print(f"Grid ICON preparado: {indices.size:,} pontos na área de estudo; arquivo de coordenadas={COORDS_FILE}")

def extract(path):
 ds=xr.open_dataset(path,engine="cfgrib",decode_timedelta=False,backend_kwargs={"indexpath":""})
 try:
  chosen=None
  for name,da in ds.data_vars.items():
   arr=np.asarray(da.squeeze(drop=True).values)
   if arr.ndim==1:
    chosen=(da,arr); break
  if chosen is None: raise ValueError(f"ICON não contém campo 1D não estruturado: {list(ds.data_vars)}")
  da,raw_values=chosen
  values=np.asarray(raw_values,dtype=float).reshape(-1)
  with np.load(COORDS_FILE) as coords:
   indices=coords["indices"]; lon=coords["lon"]; lat=coords["lat"]; total=int(coords["total"])
  if values.size!=total or lon.size!=indices.size or lat.size!=indices.size:
   raise ValueError(f"Grade ICON incoerente: valores={values.size}, totalGrid={total}, pontosÁrea={indices.size}")
  values=values[indices]
  valid=np.isfinite(values)
  if int(valid.sum())<100: raise ValueError("Poucos valores ICON válidos na área")
  return lon[valid],lat[valid],values[valid],str(da.attrs.get("units",""))
 finally:
  ds.close()

def render(lon,lat,values,key,run,step,out,units=""):
 cfg=FIELDS[key]
 valid=run+timedelta(hours=step)
 return render_all_views(
  lon,lat,values,key,cfg["label"],valid,out,
  source="DWD ICON Global Open Data • native grid",
  region_root=OUT/"regions",region_relative=out.relative_to(OUT),
  unit=units or cfg["unit"],unstructured=True,
  run_time=run,forecast_hour=step,model_label="DWD ICON Global",
 )

def render_wind(u,v,run,step,out):
 lon,lat,uv,units=extract(u); lon2,lat2,vv,_=extract(v)
 if lon.shape!=lon2.shape or not np.allclose(lon,lon2) or not np.allclose(lat,lat2):
  raise ValueError("Grades U/V do ICON não coincidem")
 speed=np.hypot(uv,vv)
 valid=run+timedelta(hours=step)
 return render_all_views(
  lon,lat,speed,"wind10m","Velocidade do vento a 10 m",valid,out,
  source="DWD ICON Global Open Data • native grid",
  region_root=OUT/"regions",region_relative=out.relative_to(OUT),
  unit=units or "m/s",wind_u=uv,wind_v=vv,unstructured=True,
  run_time=run,forecast_hour=step,model_label="DWD ICON Global",
 )

def main():
 run=latest_run(); tag=run.strftime("%Y%m%d%H"); steps=steps_for_run(run)
 report={"model":"ICON-GLOBAL","run":tag,"source":BASE,"extent":list(EXTENT),"forecastHours":steps,
         "temporalCadence":{"hourlyThrough":78,"extendedEveryHours":3,"forecastLimit":max(steps)},"generated":[],"failures":[]}
 try:
  ensure_coordinates()
 except Exception as exc:
  report["failures"].append({"field":"grid_coordinates","error":f"{type(exc).__name__}: {exc}"})
  report["generatedCount"]=0; report["failureCount"]=1
  OUT.mkdir(parents=True,exist_ok=True)
  (OUT/"generation-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
  raise SystemExit(f"Falha ao preparar grid ICON Global: {exc}")
 for step in steps:
  for key,cfg in FIELDS.items():
   try:
    path=fetch_grib(run,step,cfg["folder"],cfg["suffix"]); lon,lat,values,units=extract(path)
    out=OUT/tag/key/f"f{step:03d}.png"; written=render(lon,lat,values,key,run,step,out,units)
    report["generated"].extend(str(p) for p in written); print(f"OK {out} + {len(written)-1} recortes regionais")
   except Exception as exc:
    report["failures"].append({"hour":step,"field":key,"error":f"{type(exc).__name__}: {exc}"})
    print(f"FALHA ICON F{step:03d} {key}: {exc}")
  try:
   up=fetch_grib(run,step,WIND["u"]["folder"],WIND["u"]["suffix"])
   vp=fetch_grib(run,step,WIND["v"]["folder"],WIND["v"]["suffix"])
   out=OUT/tag/"wind10m"/f"f{step:03d}.png"; written=render_wind(up,vp,run,step,out)
   report["generated"].extend(str(p) for p in written); print(f"OK {out} + {len(written)-1} recortes regionais")
  except Exception as exc:
   report["failures"].append({"hour":step,"field":"wind10m","error":f"{type(exc).__name__}: {exc}"})
   print(f"FALHA ICON F{step:03d} wind10m: {exc}")
  for key,cfg in PRESSURE_FIELDS.items():
   try:
    level=cfg["level"]
    if key.startswith("wind"):
     up=fetch_grib(run,step,cfg["u_folder"],cfg["u_suffix"],level=level)
     vp=fetch_grib(run,step,cfg["v_folder"],cfg["v_suffix"],level=level)
     lon,lat,uvals,units=extract(up); lon2,lat2,vvals,_=extract(vp)
     if lon.shape!=lon2.shape or not np.allclose(lon,lon2) or not np.allclose(lat,lat2) or uvals.shape!=vvals.shape:
      raise ValueError(f"Grades U/V do ICON não coincidem em {level} hPa")
     values=np.hypot(uvals,vvals); wind_u=uvals; wind_v=vvals
     unit=units or cfg["unit"]
    else:
     path=fetch_grib(run,step,cfg["folder"],cfg["suffix"],level=level)
     lon,lat,values,units=extract(path); wind_u=wind_v=None; unit=units or cfg["unit"]
    valid=run+timedelta(hours=step)
    title=cfg["label"]
    out=OUT/tag/key/f"f{step:03d}.png"
    written=render_all_views(
     lon,lat,values,key,title,valid,out,
     source="DWD ICON Global Open Data • native grid",
     region_root=OUT/"regions",region_relative=out.relative_to(OUT),
     unit=unit,wind_u=wind_u,wind_v=wind_v,unstructured=True,
     run_time=run,forecast_hour=step,model_label="DWD ICON Global",
    )
    report["generated"].extend(str(p) for p in written)
    print(f"OK {out} + {len(written)-1} recortes regionais",flush=True)
   except Exception as exc:
    report["failures"].append({"hour":step,"field":key,"error":f"{type(exc).__name__}: {exc}"})
    print(f"FALHA ICON F{step:03d} {key}: {exc}",flush=True)
 OUT.mkdir(parents=True,exist_ok=True)
 report["generatedCount"]=len(report["generated"]); report["failureCount"]=len(report["failures"]); report["regionalViewsPerSourceMap"]=len(REGIONS); report["gridResolutionPreserved"]=True
 (OUT/"generation-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps({k:report[k] for k in ("run","generatedCount","failureCount")},indent=2))
 if report["generatedCount"]==0: raise SystemExit("Nenhum mapa ICON gerado; consulte o relatório")
if __name__=="__main__": main()
