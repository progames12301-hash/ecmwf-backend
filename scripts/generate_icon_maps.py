#!/usr/bin/env python3
"""Operational DWD ICON-GLOBAL maps for South America; downloads only selected fields/steps."""
from __future__ import annotations
import bz2, json, re, urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cartopy.crs as ccrs
import cartopy.feature as cfeature

EXTENT=(-90,-30,-60,15)
STEPS=[0,24,48,72,120]
BASE="https://opendata.dwd.de/weather/nwp/icon/grib"
CACHE=Path("work/icon-grib")
OUT=Path("site/maps/icon/global")
# DWD ICON-GLOBAL operational single-level fields. EU-only ICON is intentionally not used:
# it does not cover South America.
FIELDS={
 "temp2m":{"folder":"t_2m","suffix":"T_2M","label":"Temperatura a 2 m","unit":"°C","cmap":"coolwarm","vmin":-10,"vmax":40,"kind":"temp"},
 "dew2m":{"folder":"td_2m","suffix":"TD_2M","label":"Ponto de orvalho a 2 m","unit":"°C","cmap":"BrBG","vmin":-20,"vmax":30,"kind":"temp"},
 "precip":{"folder":"tot_prec","suffix":"TOT_PREC","label":"Precipitação acumulada","unit":"mm","cmap":"turbo","vmin":0,"vmax":80,"kind":"precip"},
 "cape":{"folder":"cape_con","suffix":"CAPE_CON","label":"CAPE convectiva","unit":"J/kg","cmap":"magma","vmin":0,"vmax":4000,"kind":"cape"},
}
WIND={"u":{"folder":"u_10m","suffix":"U_10M"},"v":{"folder":"v_10m","suffix":"V_10M"}}

def latest_run():
 t=datetime.now(timezone.utc)-timedelta(hours=6)
 return t.replace(hour=(t.hour//6)*6,minute=0,second=0,microsecond=0)

def links_from_index(url):
 req=urllib.request.Request(url,headers={"User-Agent":"SideralMeteorologia-map-runner/1.0"})
 with urllib.request.urlopen(req,timeout=60) as r: html=r.read().decode("utf-8","replace")
 return re.findall(r"""href=["']([^"']+\.grib2\.bz2)["']""", html, re.I)

def fetch_grib(run,step,folder,suffix):
 cycle=f"{run.hour:02d}"
 directory=f"{BASE}/{cycle}/{folder}/"
 filename=f"icon_global_icosahedral_single-level_{run:%Y%m%d%H}_{step:03d}_{suffix}.grib2.bz2"
 local_dir=CACHE/f"{run:%Y%m%d%H}"/f"{step:03d}"
 local_dir.mkdir(parents=True,exist_ok=True)
 out=local_dir/(suffix+".grib2")
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

def extract(path):
 ds=xr.open_dataset(path,engine="cfgrib",decode_timedelta=False,backend_kwargs={"indexpath":""})
 try:
  chosen=None
  for name,da in ds.data_vars.items():
   arr=np.asarray(da.squeeze(drop=True).values)
   if arr.ndim==1:
    chosen=(da,arr); break
  if chosen is None: raise ValueError(f"ICON não contém campo 1D não estruturado: {list(ds.data_vars)}")
  da,values=chosen
  lon=np.asarray(ds["longitude"].values).reshape(-1)
  lat=np.asarray(ds["latitude"].values).reshape(-1)
  if lon.size!=values.size or lat.size!=values.size:
   raise ValueError(f"Dimensões ICON incoerentes: val={values.shape}, lon={lon.shape}, lat={lat.shape}")
  lon=((lon+180)%360)-180
  mask=(lon>=EXTENT[0]-5)&(lon<=EXTENT[1]+5)&(lat>=EXTENT[2]-5)&(lat<=EXTENT[3]+5)&np.isfinite(values)
  lon,lat,values=lon[mask],lat[mask],np.asarray(values[mask],dtype=float)
  if values.size<100: raise ValueError(f"Poucos pontos ICON na área: {values.size}")
  return lon,lat,values,str(da.attrs.get("units",""))
 finally:
  ds.close()

def render(lon,lat,values,key,run,step,out):
 cfg=FIELDS[key]
 vals=np.array(values,dtype=float,copy=True)
 if cfg["kind"]=="temp" and np.nanmedian(vals)>150: vals-=273.15
 if cfg["kind"]=="precip": vals=np.maximum(vals,0)
 tri=mtri.Triangulation(lon,lat)
 # Remove long triangles caused by the longitude seam or sparse edge coverage.
 triangles=tri.triangles
 x=lon[triangles]; y=lat[triangles]
 max_edge=np.maximum.reduce([
  np.hypot(x[:,0]-x[:,1],y[:,0]-y[:,1]),
  np.hypot(x[:,1]-x[:,2],y[:,1]-y[:,2]),
  np.hypot(x[:,2]-x[:,0],y[:,2]-y[:,0])
 ])
 tri.set_mask((max_edge>2.0)|np.any(~np.isfinite(vals[triangles]),axis=1))
 fig=plt.figure(figsize=(12,8),dpi=115)
 ax=plt.axes(projection=ccrs.PlateCarree()); ax.set_extent(EXTENT,crs=ccrs.PlateCarree())
 ax.set_facecolor("#f8fafc")
 ax.add_feature(cfeature.OCEAN,facecolor="#f8fafc",zorder=0)
 ax.add_feature(cfeature.LAND,facecolor="#f2f0e8",edgecolor="#999999",linewidth=.35,zorder=1)
 im=ax.tricontourf(tri,vals,levels=24,cmap=cfg["cmap"],vmin=cfg["vmin"],vmax=cfg["vmax"],transform=ccrs.PlateCarree(),zorder=2)
 ax.coastlines("110m",linewidth=.65,color="#333333",zorder=3)
 ax.add_feature(cfeature.BORDERS,linewidth=.4,edgecolor="#666666",zorder=3)
 gl=ax.gridlines(draw_labels=True,linewidth=.25,color="#666666",alpha=.35,linestyle=":")
 gl.top_labels=False; gl.right_labels=False
 valid=run+timedelta(hours=step)
 fig.suptitle(f"ICON GLOBAL • {cfg['label']}\\nF{step:03d} | válido {valid:%Y-%m-%d %HZ}",fontsize=13,fontweight="bold",y=.97)
 cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=.045,fraction=.045); cb.set_label(cfg["unit"])
 fig.text(.995,.006,"Fonte: DWD ICON Global Open Data • Sideral Meteorologia",ha="right",fontsize=7,color="#555")
 out.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(out,bbox_inches="tight",facecolor="white"); plt.close(fig)
 if not out.exists() or out.stat().st_size<1000: raise RuntimeError(f"PNG inválido: {out}")

def render_wind(u,v,run,step,out):
 lon,lat,uv,_=extract(u); lon2,lat2,vv,_=extract(v)
 if lon.shape!=lon2.shape or not np.allclose(lon,lon2) or not np.allclose(lat,lat2):
  raise ValueError("Grades U/V do ICON não coincidem")
 cfg={"label":"Velocidade do vento a 10 m","unit":"m/s","cmap":"turbo","vmin":0,"vmax":30}
 speed=np.hypot(uv,vv)
 tri=mtri.Triangulation(lon,lat); tri.set_mask(np.any(~np.isfinite(speed[tri.triangles]),axis=1))
 fig=plt.figure(figsize=(12,8),dpi=115); ax=plt.axes(projection=ccrs.PlateCarree())
 ax.set_extent(EXTENT,crs=ccrs.PlateCarree()); ax.add_feature(cfeature.OCEAN,facecolor="#f8fafc",zorder=0)
 ax.add_feature(cfeature.LAND,facecolor="#f2f0e8",edgecolor="#999999",linewidth=.35,zorder=1)
 im=ax.tricontourf(tri,speed,levels=24,cmap=cfg["cmap"],vmin=cfg["vmin"],vmax=cfg["vmax"],transform=ccrs.PlateCarree(),zorder=2)
 ax.coastlines("110m",linewidth=.65,color="#333333",zorder=3); ax.add_feature(cfeature.BORDERS,linewidth=.4,edgecolor="#666666",zorder=3)
 gl=ax.gridlines(draw_labels=True,linewidth=.25,color="#666666",alpha=.35,linestyle=":"); gl.top_labels=False; gl.right_labels=False
 valid=run+timedelta(hours=step)
 fig.suptitle(f"ICON GLOBAL • {cfg['label']}\\nF{step:03d} | válido {valid:%Y-%m-%d %HZ}",fontsize=13,fontweight="bold",y=.97)
 cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=.045,fraction=.045); cb.set_label(cfg["unit"])
 fig.text(.995,.006,"Fonte: DWD ICON Global Open Data • Sideral Meteorologia",ha="right",fontsize=7,color="#555")
 out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight",facecolor="white"); plt.close(fig)
 if not out.exists() or out.stat().st_size<1000: raise RuntimeError(f"PNG inválido: {out}")

def main():
 run=latest_run(); tag=run.strftime("%Y%m%d%H")
 report={"model":"ICON-GLOBAL","run":tag,"source":BASE,"extent":list(EXTENT),"forecastHours":STEPS,"generated":[],"failures":[]}
 for step in STEPS:
  for key,cfg in FIELDS.items():
   try:
    path=fetch_grib(run,step,cfg["folder"],cfg["suffix"]); lon,lat,values,units=extract(path)
    out=OUT/tag/key/f"f{step:03d}.png"; render(lon,lat,values,key,run,step,out)
    report["generated"].append(str(out)); print(f"OK {out}")
   except Exception as exc:
    report["failures"].append({"hour":step,"field":key,"error":f"{type(exc).__name__}: {exc}"})
    print(f"FALHA ICON F{step:03d} {key}: {exc}")
  try:
   up=fetch_grib(run,step,WIND["u"]["folder"],WIND["u"]["suffix"])
   vp=fetch_grib(run,step,WIND["v"]["folder"],WIND["v"]["suffix"])
   out=OUT/tag/"wind10m"/f"f{step:03d}.png"; render_wind(up,vp,run,step,out)
   report["generated"].append(str(out)); print(f"OK {out}")
  except Exception as exc:
   report["failures"].append({"hour":step,"field":"wind10m","error":f"{type(exc).__name__}: {exc}"})
   print(f"FALHA ICON F{step:03d} wind10m: {exc}")
 OUT.mkdir(parents=True,exist_ok=True)
 report["generatedCount"]=len(report["generated"]); report["failureCount"]=len(report["failures"])
 (OUT/"generation-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps({k:report[k] for k in ("run","generatedCount","failureCount")},indent=2))
 if report["generatedCount"]==0: raise SystemExit("Nenhum mapa ICON gerado; consulte o relatório")
if __name__=="__main__": main()
