#!/usr/bin/env python3
"""GEFS ensemble maps for South America. Outputs mean, spread and control independently."""
from __future__ import annotations
import json
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

EXTENT=(-90,-30,-60,15)
HOURS=list(range(0,145,3))+list(range(150,241,6))  # cadência alinhada aos mapas ECMWF
MEMBERS={"mean":"mean","spread":"spr","control":"c00"}
FIELDS={
 "temp2m":("TMP:2 m","Temperatura a 2 m","°C","coolwarm",-10,40,"temp"),
 "precip":("APCP:surface","Precipitação acumulada","mm","turbo",0,80,"precip"),
 "wind10m":(None,"Velocidade do vento a 10 m","m/s","turbo",0,30,"wind"),
 "mslp":("PRMSL:mean sea level","Pressão ao nível do mar","hPa","viridis",940,1040,"pressure"),
 "gh500":("HGT:500 mb","Altura geopotencial em 500 hPa","m","viridis",4800,6000,"height"),
}
OUT=Path("site/maps/gefs")

def latest_run():
 t=datetime.now(timezone.utc)-timedelta(hours=8)
 return t.replace(hour=(t.hour//6)*6,minute=0,second=0,microsecond=0)

def extract(ds):
 for name,da in ds.data_vars.items():
  vals=np.asarray(da.squeeze(drop=True).values)
  if vals.ndim==2:
   lat=np.asarray(ds["latitude"].values); lon=np.asarray(ds["longitude"].values)
   if lat.ndim!=1 or lon.ndim!=1 or vals.shape!=(lat.size,lon.size):
    continue
   lon=((lon+180)%360)-180
   ix=np.argsort(lon); lon=lon[ix]; vals=vals[:,ix]
   latmask=(lat>=EXTENT[2]-2)&(lat<=EXTENT[3]+2)
   lonmask=(lon>=EXTENT[0]-2)&(lon<=EXTENT[1]+2)
   return lon[lonmask],lat[latmask],np.asarray(vals[latmask,:][:,lonmask],dtype=float),str(da.attrs.get("units",""))
 raise ValueError(f"Não encontrei uma grade lat/lon 2D: {list(ds.data_vars)}")

def load_one(h,search):
 result=h.xarray(search)
 datasets=result if isinstance(result,list) else [result]
 errors=[]
 for ds in datasets:
  try:
   if not hasattr(ds,"data_vars"):
    errors.append(f"retorno não é Dataset: {type(ds).__name__}")
    continue
   return extract(ds)
  except Exception as exc:
   errors.append(f"{type(exc).__name__}: {exc}")
  finally:
   try: ds.close()
   except Exception: pass
 raise ValueError(f"Nenhum Dataset GRIB válido para {search!r}: {' | '.join(errors)}")

def render(lon,lat,vals,key,member,step,run,out,units="",wind_u=None,wind_v=None):
 label,unit,cmap,vmin,vmax,convert=FIELDS[key][1:]
 valid=run+timedelta(hours=step)
 title=f"{label}\nF{step:03d} Valid: {valid:%a %Y-%m-%d %HZ}"
 render_sideral_map(
  lon,lat,vals,key,title,valid,out,
  source=f"NOAA/NCEP GEFS {member.upper()} • 0.5° Open Data",
  unit=units or unit,wind_u=wind_u,wind_v=wind_v,
)

def main():
 run=latest_run(); run_tag=run.strftime("%Y%m%d%H")
 report={"model":"GEFS","run":run_tag,"source":"NOAA/NCEP GEFS via Herbie/AWS","product":"atmos.5",
         "ensembleProducts":{"mean":"ensemble mean","spr":"ensemble spread","c00":"control"},
         "forecastHours":HOURS,"extent":list(EXTENT),"generated":[],"failures":[],"skipped":[],"members":{}}
 for member,member_arg in MEMBERS.items():
  count=0
  for hour in HOURS:
   try:
    h=Herbie(run.replace(tzinfo=None),model="gefs",product="atmos.5",member=member_arg,fxx=hour,priority=["aws"])
    for key,(search,*_) in FIELDS.items():
     if key=="precip" and hour==0:
      report["skipped"].append({"member":member,"hour":hour,"field":key,"reason":"F000 não tem precipitação acumulada desde o tempo inicial"})
      continue
     try:
      wind_u=wind_v=None
      if key=="wind10m":
       lon,lat,wind_u,units=load_one(h,"UGRD:10 m")
       lon2,lat2,wind_v,_=load_one(h,"VGRD:10 m")
       if not (np.allclose(lon,lon2) and np.allclose(lat,lat2) and wind_u.shape==wind_v.shape):
        raise ValueError("Grades U/V do GEFS não coincidem")
       vals=np.hypot(wind_u,wind_v)
      else:
       lon,lat,vals,units=load_one(h,search)
      out=OUT/"atmos.5"/member/run_tag/key/f"f{hour:03d}.png"
      render(lon,lat,vals,key,member,hour,run,out,units=units,wind_u=wind_u,wind_v=wind_v)
      report["generated"].append(str(out)); count+=1; print(f"OK {out}")
     except Exception as exc:
      report["failures"].append({"member":member,"hour":hour,"field":key,"error":f"{type(exc).__name__}: {exc}"})
      print(f"FALHA GEFS {member} F{hour:03d} {key}: {exc}")
   except Exception as exc:
    report["failures"].append({"member":member,"hour":hour,"field":"cycle","error":f"{type(exc).__name__}: {exc}"})
    print(f"FALHA GEFS {member} F{hour:03d}: {exc}")
  report["members"][member]={"herbieMember":member_arg,"mapsGenerated":count}
 OUT.mkdir(parents=True,exist_ok=True)
 report["generatedCount"]=len(report["generated"]); report["failureCount"]=len(report["failures"])
 (OUT/"generation-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps({k:report[k] for k in ("run","generatedCount","failureCount","members")},indent=2))
 if report["generatedCount"]==0: raise SystemExit("Nenhum mapa GEFS foi gerado; consulte o relatório")
if __name__=="__main__": main()
