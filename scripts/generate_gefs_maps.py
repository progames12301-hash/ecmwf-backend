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

EXTENT=(-90,-30,-60,15)
HOURS=[0,24,48,72,96,120]
MEMBERS={"mean":"mean","spread":"spr","control":"c00"}
FIELDS={
 "temp2m":("TMP:2 m","Temperatura a 2 m","°C","coolwarm",-10,40,"temp"),
 "precip":("APCP:surface","Precipitação acumulada","mm","turbo",0,80,"precip"),
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

def render(lon,lat,vals,key,member,step,run,out):
 label,unit,cmap,vmin,vmax,convert=FIELDS[key][1:]
 vals=np.array(vals,dtype=float,copy=True)
 if convert=="temp" and np.nanmedian(vals)>150: vals-=273.15
 if convert=="pressure" and np.nanmedian(vals)>2000: vals/=100
 fig=plt.figure(figsize=(12,8),dpi=115)
 ax=plt.axes(projection=ccrs.PlateCarree())
 ax.set_extent(EXTENT,crs=ccrs.PlateCarree())
 ax.set_facecolor("#f8fafc")
 ax.add_feature(cfeature.OCEAN,facecolor="#f8fafc",zorder=0)
 ax.add_feature(cfeature.LAND,facecolor="#f2f0e8",edgecolor="#999999",linewidth=.35,zorder=1)
 im=ax.pcolormesh(lon,lat,vals,transform=ccrs.PlateCarree(),shading="auto",cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True,zorder=2)
 ax.coastlines("110m",linewidth=.65,color="#333333",zorder=3)
 ax.add_feature(cfeature.BORDERS,linewidth=.4,edgecolor="#666666",zorder=3)
 gl=ax.gridlines(draw_labels=True,linewidth=.25,color="#666666",alpha=.35,linestyle=":")
 gl.top_labels=False; gl.right_labels=False
 valid=run+timedelta(hours=step)
 fig.suptitle(f"GEFS • {label} • {member.upper()}\nF{step:03d} | válido {valid:%Y-%m-%d %HZ}",fontsize=13,fontweight="bold",y=.97)
 cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=.045,fraction=.045); cb.set_label(unit)
 fig.text(.995,.006,"Fonte: NOAA/NCEP GEFS • Sideral Meteorologia",ha="right",fontsize=7,color="#555")
 out.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(out,bbox_inches="tight",facecolor="white"); plt.close(fig)
 if not out.exists() or out.stat().st_size<1000: raise RuntimeError(f"PNG inválido: {out}")

def main():
 run=latest_run(); run_tag=run.strftime("%Y%m%d%H")
 report={"model":"GEFS","run":run_tag,"source":"NOAA/NCEP GEFS via Herbie/AWS","product":"atmos.5",
         "ensembleProducts":{"mean":"ensemble mean","spr":"ensemble spread","c00":"control"},
         "forecastHours":HOURS,"extent":list(EXTENT),"generated":[],"failures":[],"members":{}}
 for member,member_arg in MEMBERS.items():
  count=0
  for hour in HOURS:
   try:
    h=Herbie(run,model="gefs",product="atmos.5",member=member_arg,fxx=hour,priority=["aws"])
    for key,(search,*_) in FIELDS.items():
     try:
      ds=h.xarray(search)
      try: lon,lat,vals,units=extract(ds)
      finally:
       try: ds.close()
       except Exception: pass
      out=OUT/"atmos.5"/member/run_tag/key/f"f{hour:03d}.png"
      render(lon,lat,vals,key,member,hour,run,out)
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
