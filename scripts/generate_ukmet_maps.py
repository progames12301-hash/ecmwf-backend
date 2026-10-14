#!/usr/bin/env python3
"""UKMET GRIB2 map runner. Uses files/URLs supplied by the account's authorized Met Office DataHub subscription."""
from __future__ import annotations
import bz2, json, os, re, shutil, urllib.request, zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import xarray as xr
import cfgrib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

EXTENT=(-90,-30,-60,15)
OUT=Path("site/maps/ukmet")
CACHE=Path("work/ukmet")
MAX_MAPS=int(os.environ.get("UKMET_MAX_MAPS","250"))

def safe_slug(value):
 s=re.sub(r"[^a-zA-Z0-9_-]+","_",str(value)).strip("_").lower()
 return (s or "field")[:70]

def download_resource(url,api_key,dst):
 if urlparse(url).scheme!="https": raise ValueError(f"A URL do UKMET deve usar HTTPS: {url}")
 req=urllib.request.Request(url,headers={"apikey":api_key,"User-Agent":"SideralMeteorologia-UKMET-runner/1.0"})
 with urllib.request.urlopen(req,timeout=240) as r:
  body=r.read(); content_type=r.headers.get("Content-Type","")
 if "json" in content_type.lower() or body[:1] in (b"{",b"["):
  try: data=json.loads(body.decode("utf-8"))
  except Exception: data=None
  found=[]
  def walk(obj):
   if isinstance(obj,dict):
    for k,v in obj.items():
     if isinstance(v,str) and v.startswith("https://") and any(ext in v.lower() for ext in (".grib",".grb",".zip","download")):
      found.append(v)
     else: walk(v)
   elif isinstance(obj,list):
    for v in obj: walk(v)
  if data is not None: walk(data)
  if not found: raise ValueError(f"O endpoint retornou JSON mas não apresentou link de arquivo GRIB/ZIP. Content-Type={content_type}")
  downloaded=[]
  for i,link in enumerate(dict.fromkeys(found)):
   downloaded.extend(download_resource(link,api_key,dst/f"resource_{i}"))
  return downloaded
 dst.parent.mkdir(parents=True,exist_ok=True)
 if zipfile.is_zipfile(__import__("io").BytesIO(body)):
  dst.mkdir(parents=True,exist_ok=True); result=[]
  import io
  with zipfile.ZipFile(io.BytesIO(body)) as z:
   for member in z.namelist():
    if member.lower().endswith((".grib2",".grb2",".grib",".grb")):
     target=dst/safe_slug(Path(member).stem)
     target=target.with_suffix(".grib2")
     target.write_bytes(z.read(member)); result.append(target)
  if not result: raise ValueError("ZIP do UKMET sem arquivos GRIB2 reconhecidos")
  return result
 if body[:3]==b"BZh": body=bz2.decompress(body)
 if not body.startswith(b"GRIB"):
  raise ValueError(f"Resposta do DataHub não é GRIB2 nem ZIP (Content-Type={content_type}); confira a URL configurada")
 out=dst
 if out.suffix not in (".grib",".grb",".grib2",".grb2"): out=out.with_suffix(".grib2")
 out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(body)
 return [out]

def normalize_grid(ds,da):
 vals=np.asarray(da.squeeze(drop=True).values,dtype=float)
 if vals.ndim!=2: return None
 lat=np.asarray(ds["latitude"].values); lon=np.asarray(ds["longitude"].values)
 if lat.ndim==1 and lon.ndim==1 and vals.shape==(lat.size,lon.size):
  lon=((lon+180)%360)-180; ix=np.argsort(lon); lon=lon[ix]; vals=vals[:,ix]
  lm=(lat>=EXTENT[2]-2)&(lat<=EXTENT[3]+2); xm=(lon>=EXTENT[0]-2)&(lon<=EXTENT[1]+2)
  return lon[xm],lat[lm],vals[lm,:][:,xm]
 if lat.ndim==2 and lon.ndim==2 and vals.shape==lat.shape:
  lon=((lon+180)%360)-180
  mask=(lon>=EXTENT[0]-2)&(lon<=EXTENT[1]+2)&(lat>=EXTENT[2]-2)&(lat<=EXTENT[3]+2)
  return lon,lat,np.where(mask,vals,np.nan)
 return None

def render(ds,da,step_hours,run_label,field_name,out):
 grid=normalize_grid(ds,da)
 if grid is None: raise ValueError(f"Grade incompatível para {field_name}: dims={da.dims}, coords={list(ds.coords)}")
 lon,lat,vals=grid
 unit=str(da.attrs.get("units",""))
 short=str(da.attrs.get("GRIB_shortName",da.name or field_name))
 if short.lower() in ("2t","t2m","t","2d","d2m") and np.nanmedian(vals)>150: vals=vals-273.15; unit="°C"
 if short.lower() in ("msl","prmsl") and np.nanmedian(vals)>2000: vals=vals/100; unit="hPa"
 label=str(da.attrs.get("long_name",da.attrs.get("GRIB_name",field_name)))
 vmax=np.nanpercentile(vals,98) if np.isfinite(vals).any() else 1
 vmin=np.nanpercentile(vals,2) if np.isfinite(vals).any() else 0
 cmap="turbo" if any(t in (short+" "+label).lower() for t in ("tp","precip","rain")) else "coolwarm" if "t" in short.lower() else "viridis"
 fig=plt.figure(figsize=(12,8),dpi=115); ax=plt.axes(projection=ccrs.PlateCarree())
 ax.set_extent(EXTENT,crs=ccrs.PlateCarree()); ax.set_facecolor("#f8fafc")
 ax.add_feature(cfeature.OCEAN,facecolor="#f8fafc",zorder=0)
 ax.add_feature(cfeature.LAND,facecolor="#f2f0e8",edgecolor="#999999",linewidth=.35,zorder=1)
 im=ax.pcolormesh(lon,lat,vals,transform=ccrs.PlateCarree(),shading="auto",cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True,zorder=2)
 ax.coastlines("110m",linewidth=.65,color="#333333",zorder=3); ax.add_feature(cfeature.BORDERS,linewidth=.4,edgecolor="#666666",zorder=3)
 gl=ax.gridlines(draw_labels=True,linewidth=.25,color="#666666",alpha=.35,linestyle=":"); gl.top_labels=False; gl.right_labels=False
 fig.suptitle(f"UKMET • {label}\\n{run_label} • F{step_hours:03d}",fontsize=13,fontweight="bold",y=.97)
 cb=fig.colorbar(im,ax=ax,orientation="horizontal",pad=.045,fraction=.045); cb.set_label(unit)
 fig.text(.995,.006,"Fonte: Met Office Weather DataHub (credencial do assinante) • Sideral Meteorologia",ha="right",fontsize=7,color="#555")
 out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,bbox_inches="tight",facecolor="white"); plt.close(fig)
 if not out.exists() or out.stat().st_size<1000: raise RuntimeError(f"PNG inválido: {out}")

def main():
 api_key=os.environ.get("METOFFICE_API_KEY","").strip()
 raw_urls=os.environ.get("METOFFICE_GRIB_URLS","").strip()
 tag=datetime.now(timezone.utc).strftime("%Y%m%d%H")
 OUT.mkdir(parents=True,exist_ok=True)
 report={"model":"UKMET","createdUTC":datetime.now(timezone.utc).isoformat(),"source":"Met Office Weather DataHub",
         "status":"pending","extent":list(EXTENT),"generated":[],"failures":[]}
 if not api_key or not raw_urls:
  report.update(status="skipped_missing_credentials",
   reason="Configure os secrets METOFFICE_API_KEY e METOFFICE_GRIB_URLS com a chave DataHub e URL(s) direta(s) de arquivos/ZIP GRIB2 do produto global UKMET contratado.")
  (OUT/"generation-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
  print("AVISO: UKMET não baixado. Faltam METOFFICE_API_KEY e/ou METOFFICE_GRIB_URLS; nenhum mapa fictício foi criado.")
  return
 CACHE.mkdir(parents=True,exist_ok=True)
 urls=[u.strip() for u in raw_urls.splitlines() if u.strip()]
 files=[]
 for i,url in enumerate(urls):
  try:
   files.extend(download_resource(url,api_key,CACHE/f"download_{i}.grib2"))
  except Exception as exc:
   report["failures"].append({"urlHost":urlparse(url).netloc,"error":f"{type(exc).__name__}: {exc}"})
   print(f"FALHA download UKMET {urlparse(url).netloc}: {exc}")
 map_count=0
 for file in files:
  try: datasets=cfgrib.open_datasets(str(file),backend_kwargs={"indexpath":""})
  except Exception as exc:
   report["failures"].append({"file":file.name,"error":f"cfgrib: {type(exc).__name__}: {exc}"}); continue
  for gi,ds in enumerate(datasets):
   try:
    for name,da in ds.data_vars.items():
     if map_count>=MAX_MAPS: break
     steps=range(da.sizes["step"]) if "step" in da.dims else [None]
     for ix in steps:
      if map_count>=MAX_MAPS: break
      selected=da.isel(step=ix) if ix is not None else da
      raw=selected.coords.get("step",None)
      step=0
      if raw is not None:
       value=np.asarray(raw.values).reshape(-1)[0]
       try: step=int(value/np.timedelta64(1,"h")) if isinstance(value,np.timedelta64) else int(value)
       except Exception: step=0
      level=selected.attrs.get("GRIB_level",selected.attrs.get("GRIB_typeOfLevel",""))
      slug=safe_slug(f"{name}_{level}_{gi}")
      out=OUT/tag/slug/f"f{step:03d}.png"
      try:
       render(ds,selected,step,tag,name,out)
       report["generated"].append(str(out)); map_count+=1; print(f"OK {out}")
      except Exception as exc:
       report["failures"].append({"file":file.name,"field":name,"step":step,"error":f"{type(exc).__name__}: {exc}"})
   finally:
    try: ds.close()
    except Exception: pass
  if map_count>=MAX_MAPS: break
 report["generatedCount"]=map_count; report["failureCount"]=len(report["failures"])
 report["status"]="complete" if map_count>0 else "failed"
 (OUT/"generation-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 print(json.dumps({k:report.get(k) for k in ("status","generatedCount","failureCount","reason")},indent=2,ensure_ascii=False))
 if map_count==0: raise SystemExit("Nenhum mapa UKMET gerado; verifique a assinatura, URLs e arquivos GRIB2")

if __name__=="__main__": main()
