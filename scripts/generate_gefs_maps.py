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
from sideral_map_style import render_all_views, REGIONS, REGIONAL_FORECAST_HOURS

EXTENT=(-90,-20,-60,15)
HOURS=list(range(0,241,3))+list(range(246,385,6))  # cadência GEFS: 3 h até F240, depois 6 h até F384
MEMBERS={"mean":"mean","spread":"spr","control":"c00"}
FIELDS={
 "temp2m":("TMP:2 m","Temperatura a 2 m","°C","coolwarm",-10,40,"temp"),
 "dew2m":("DPT:2 m","Ponto de orvalho a 2 m","°C","BrBG",-20,30,"temp"),
 "wind10m":(None,"Velocidade do vento a 10 m","km h⁻¹","viridis",0,80,"wind"),
 "gust10m":("GUST:surface","Rajada máxima a 10 m","km h⁻¹","viridis",0,80,"wind"),
 "precip":("APCP:surface","Precipitação acumulada","mm","turbo",0,500,"precip"),
 "mslp":("PRMSL:mean sea level","Pressão ao nível do mar","hPa","viridis",940,1040,"pressure"),
 "cloud":("TCDC:entire atmosphere","Cobertura total de nuvens","%","Greys",0,100,"cloud"),
 "tcwv":("PWAT:entire atmosphere","Água precipitável","mm","GnBu",0,70,"tcwv"),
 "cape":("CAPE:surface","CAPE superficial","J kg⁻¹","magma",0,6000,"cape"),
 "t850":("TMP:850 mb","Temperatura em 850 hPa","°C","coolwarm",-30,30,"temp"),
 "wind850":("UGRD:850 mb","Vento em 850 hPa","km h⁻¹","viridis",0,120,"wind"),
 "gh500":("HGT:500 mb","Altura geopotencial em 500 hPa","m","viridis",4800,6000,"height"),
 "wind500":("UGRD:500 mb","Vento em 500 hPa","km h⁻¹","viridis",0,120,"wind"),
 "wind250":("UGRD:250 mb","Vento em 250 hPa","km h⁻¹","viridis",0,120,"wind"),
 "rh2m":("RH:2 m","Umidade relativa a 2 m","%","YlGnBu",0,100,"humidity"),
 "dew850":("DPT:850 mb","Ponto de orvalho em 850 hPa","°C","BrBG",-40,25,"temp"),
 "rh850":("RH:850 mb","Umidade relativa em 850 hPa","%","YlGnBu",0,100,"humidity"),
 "t700":("TMP:700 mb","Temperatura em 700 hPa","°C","coolwarm",-40,25,"temp"),
 "wind700":("UGRD:700 mb","Vento em 700 hPa","km h⁻¹","viridis",0,120,"wind"),
 "gh700":("HGT:700 mb","Altura geopotencial em 700 hPa","m","viridis",2000,3500,"height"),
 "wind300":("UGRD:300 mb","Vento em 300 hPa","km h⁻¹","viridis",0,180,"wind"),
 "gh300":("HGT:300 mb","Altura geopotencial em 300 hPa","m","viridis",8500,10000,"height"),
}
WIND_LEVELS={"wind10m":"10 m","wind850":"850 mb","wind500":"500 mb","wind250":"250 mb","wind700":"700 mb","wind300":"300 mb"}

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
 return render_all_views(
  lon,lat,vals,key,label,valid,out,
  source=f"NOAA/NCEP GEFS {member.upper()} • 0.5° native grid",
  region_root=OUT/"regions",region_relative=out.relative_to(OUT),
  unit=units or unit,wind_u=wind_u,wind_v=wind_v,
  run_time=run,forecast_hour=step,model_label=f"GEFS {member.upper()}",
 )

def main():
 run=latest_run(); run_tag=run.strftime("%Y%m%d%H")
 report={"model":"GEFS","run":run_tag,"source":"NOAA/NCEP GEFS via Herbie/AWS","product":"atmos.5",
         "ensembleProducts":{"mean":"ensemble mean","spread":"ensemble spread","control":"control"},
         "forecastHours":HOURS,"extent":list(EXTENT),"generated":[],"failures":[],"skipped":[],"members":{},
         "fieldCount":len(FIELDS),"viewCount":len(REGIONS)+1,"regionalViewsPerSourceMap":len(REGIONS),
         "regionalForecastHours":list(REGIONAL_FORECAST_HOURS),
         "jsonSidecars":True,"jsonSchema":"sideral-native-grid-v1",
         "regions":{k:{"label":v["label"],"extent":list(v["extent"])} for k,v in REGIONS.items()}}
 for member,member_arg in MEMBERS.items():
  count=0
  for hour in HOURS:
   try:
    h=Herbie(run.replace(tzinfo=None),model="gefs",product="atmos.5",member=member_arg,fxx=hour,priority=["aws"])
    for key,(search,*_) in FIELDS.items():
     if key=="precip" and hour==0:
      report["skipped"].append({"member":member,"hour":hour,"field":key,
                                "reason":"F000 não tem precipitação acumulada desde o tempo inicial"})
      continue
     try:
      wind_u=wind_v=None
      if key in WIND_LEVELS:
       level=WIND_LEVELS[key]
       lon,lat,wind_u,units=load_one(h,f"UGRD:{level}")
       lon2,lat2,wind_v,_=load_one(h,f"VGRD:{level}")
       if not (np.allclose(lon,lon2) and np.allclose(lat,lat2) and wind_u.shape==wind_v.shape):
        raise ValueError(f"Grades U/V do GEFS não coincidem para {level}")
       vals=np.hypot(wind_u,wind_v)
      else:
       lon,lat,vals,units=load_one(h,search)
      out=OUT/"atmos.5"/member/run_tag/key/f"f{hour:03d}.png"
      written=render(lon,lat,vals,key,member,hour,run,out,units=units,wind_u=wind_u,wind_v=wind_v)
      report["generated"].extend(str(p) for p in written); count+=len(written)
      print(f"OK {out} + {len(written)-1} recortes regionais",flush=True)
     except FileNotFoundError as exc:
      # Some GEFS surface parameters are not present in every long-range subfile.
      # Record the native-source absence explicitly instead of failing the whole run or inventing values.
      report["skipped"].append({"member":member,"hour":hour,"field":key,
                                "reason":f"Parâmetro/arquivo não publicado pelo GEFS nesta validade: {exc}"})
      print(f"PULADO GEFS {member} F{hour:03d} {key}: campo não publicado nesta validade",flush=True)
     except Exception as exc:
      report["failures"].append({"member":member,"hour":hour,"field":key,"error":f"{type(exc).__name__}: {exc}"})
      print(f"FALHA GEFS {member} F{hour:03d} {key}: {exc}",flush=True)
   except Exception as exc:
    report["failures"].append({"member":member,"hour":hour,"field":"cycle","error":f"{type(exc).__name__}: {exc}"})
    print(f"FALHA GEFS {member} F{hour:03d}: {exc}",flush=True)
  report["members"][member]={"herbieMember":member_arg,"mapsGeneratedIncludingRegions":count}
 OUT.mkdir(parents=True,exist_ok=True)
 report["generatedCount"]=len(report["generated"]); report["failureCount"]=len(report["failures"])
 report["regionalViewsPerSourceMap"]=len(REGIONS); report["gridResolutionPreserved"]=True
 (OUT/"generation-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps({k:report[k] for k in ("run","generatedCount","failureCount","members")},indent=2),flush=True)
 if report["generatedCount"]==0: raise SystemExit("Nenhum mapa GEFS foi gerado; consulte o relatório")

if __name__=="__main__": main()
