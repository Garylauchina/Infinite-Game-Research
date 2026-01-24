from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os, json

app = FastAPI()

BASE = Path("/root/Infinite-Game-Research")
RUN_DIR = BASE / "runs" / "continuous_world"

@app.get("/api/runs")
def list_runs():
    if not RUN_DIR.exists():
        return JSONResponse({"runs": []})
    runs = sorted([p.name for p in RUN_DIR.iterdir() if p.is_dir()], reverse=True)
    return JSONResponse({"runs": runs})

@app.get("/api/stream/{run_id}")
def get_stream(run_id: str):
    p = RUN_DIR / run_id / "stream.json"
    if not p.exists():
        return JSONResponse({"error": "stream not found", "run_id": run_id}, status_code=404)
    return FileResponse(str(p), media_type="application/json")

@app.get("/api/latest/{run_id}")
def get_latest(run_id: str):
    p = RUN_DIR / run_id / "latest_state.json"
    if not p.exists():
        return JSONResponse({"error": "latest_state not found", "run_id": run_id}, status_code=404)
    return FileResponse(str(p), media_type="application/json")

app.mount("/", StaticFiles(directory=str(BASE / "web" / "static"), html=True), name="static")
