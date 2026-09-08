import datetime as dt
import os
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(title="Sideral ECMWF Backend", version="1.0.0")
CACHE = Path(os.getenv("ECMWF_CACHE_DIR", "/tmp/ecmwf-cache")); CACHE.mkdir(parents=True, exist_ok=True)
RESOLUTION = os.getenv("ECMWF_RESOLUTION", "0p1")
_lock = threading.Lock()

@app.get("/health")
def health():
    return {"status":"ok", "service":"sideral-ecmwf", "resolution":RESOLUTION}

@app.get("/api/ecmwf/status")
def status():
    return {"status":"ok", "model":"ECMWF IFS HRES", "resolution":"~9 km (0.1°)", "cache_files":len(list(CACHE.glob("*.grib")))}

def latest_run():
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    cycle = 12 if now.hour >= 12 else 0
    run = now.replace(hour=cycle)
    if cycle == 0 and now.hour < 6: run -= dt.timedelta(days=1)
    return run

def download_grib(run, step, target):
    from ecmwf.opendata import Client
    with _lock:
        if target.exists() and target.stat().st_size: return
        tmp = target.with_suffix(".part")
        Client(source="ecmwf", model="ifs", resol=RESOLUTION, infer_stream_keyword=True).retrieve(
            date=run.strftime("%Y%m%d"), time=run.hour, step=step, type="fc", stream="oper",
            levtype="sfc", param="msl,2t,10u,10v,tp", target=str(tmp))
        tmp.replace(target)

@app.get("/api/ecmwf/latest")
def latest(step: int = 0):
    if step < 0 or step > 240 or step % 3: raise HTTPException(400, "step inválido; use intervalos de 3 h")
    run = latest_run(); name=f"ifs_{run:%Y%m%d%H}_f{step:03d}.grib"; path=CACHE/name
    try: download_grib(run, step, path)
    except Exception as exc: raise HTTPException(502, "ECMWF indisponível no momento") from exc
    return {"model":"ECMWF IFS HRES", "resolution":"~9 km", "run":run.isoformat(), "step":step, "file_url":f"/api/ecmwf/file/{name}"}

@app.get("/api/ecmwf/file/{name}")
def file(name: str):
    if Path(name).name != name or not name.endswith(".grib"): raise HTTPException(400, "arquivo inválido")
    path=CACHE/name
    if not path.exists(): raise HTTPException(404, "arquivo não encontrado")
    return FileResponse(path, media_type="application/octet-stream", filename=name)
