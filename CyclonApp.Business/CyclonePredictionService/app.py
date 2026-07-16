"""
app.py
──────
FastAPI wrapper around the trained CyclonePINN, plus async job endpoints
for the field-solving physics-guided model (velocity/pressure fields).

EXISTING CONTRACT (unchanged):
    POST {base_url}/predict
    body: PredictionRequest  (FlowRateCFM, InletLineSizeIn, OperatingTempC,
                               OperatingPressKPa, GasType, ParticleSizeMicron,
                               ParticleDensityKgm3, EffectiveTurns,
                               InletHeightRatio, InletWidthRatio, OutletDiamRatio)
    response: PredictionResponse (PredictedEfficiency, PredictedPressureDropPa)

NEW CONTRACT (field-solving model):
    POST {base_url}/predict_field/start
    body: PredictFieldStartRequest (geometry in mm + process conditions)
    response: PredictFieldStartResponse (JobId, Status="running")

    GET {base_url}/predict_field/status/{job_id}
    response: PredictFieldStatusResponse (JobId, Status, ErrorMessage, Result)

Why async: unlike CyclonePINN, CycloneFieldPINN is trained fresh per
geometry/operating-point (see field_train.py docstring) — a full physics
solve takes real minutes, not milliseconds, so this cannot be a synchronous
request the way /predict is. The client starts a job, polls status.

Known production limitation, stated plainly: the job store below is an
in-process dict. It does not survive a service restart and does not work
across multiple uvicorn worker processes. Fine for a single-worker
deployment; if this service is ever scaled to multiple workers, the job
store needs to move to something shared (Redis, a DB table) — not done
here because it isn't needed yet and would be an unnecessary abstraction
for the current single-worker deployment.

C#'s HttpClient.PostAsJsonAsync/ReadFromJsonAsync are called without custom
JsonSerializerOptions in CyclonePredictionRepository.cs, so System.Text.Json
uses its *default* options: exact, case-sensitive PascalCase property names,
no camelCase conversion. All Pydantic models below mirror that via field
aliases, consistent with the existing /predict contract.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import threading
import time
import uuid
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from model import CyclonePINN, FeatureScaler, RAW_FEATURES
from dataset import FEATURE_RANGES
from physics import gas_type_to_onehot

from field_physics import geometry_from_dimensions_mm, fluid_properties, inlet_velocity_ms
from field_model import evaluate_grid
from field_train import train_field_model

app = FastAPI(title="Cyclone Prediction Service (Physics-Informed)")

_model: CyclonePINN | None = None
_scaler: FeatureScaler | None = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    flow_rate_cfm: float = Field(alias="FlowRateCFM")
    inlet_line_size_in: float = Field(alias="InletLineSizeIn")
    operating_temp_c: float = Field(alias="OperatingTempC")
    operating_press_kpa: float = Field(alias="OperatingPressKPa")
    gas_type: str = Field(alias="GasType", default="Air")
    particle_size_micron: float = Field(alias="ParticleSizeMicron")
    particle_density_kgm3: float = Field(alias="ParticleDensityKgm3")
    effective_turns: float = Field(alias="EffectiveTurns")
    inlet_height_ratio: float = Field(alias="InletHeightRatio")
    inlet_width_ratio: float = Field(alias="InletWidthRatio")
    outlet_diam_ratio: float = Field(alias="OutletDiamRatio")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    predicted_efficiency: float = Field(alias="PredictedEfficiency")
    predicted_pressure_drop_pa: float = Field(alias="PredictedPressureDropPa")


_ALIAS_LOOKUP = {f.alias.lower(): name for name, f in PredictionRequest.model_fields.items()}


def _case_insensitive_parse(payload: dict) -> PredictionRequest:
    """Fallback path: match incoming keys to expected fields ignoring case,
    in case the .NET side's JSON casing ever changes."""
    normalized = {}
    for k, v in payload.items():
        target = _ALIAS_LOOKUP.get(k.lower())
        if target:
            normalized[target] = v
    return PredictionRequest(**normalized)


@app.on_event("startup")
def load_model():
    global _model, _scaler
    _model = CyclonePINN()
    try:
        _model.load_state_dict(torch.load("artifacts/model.pt", map_location="cpu"))
    except FileNotFoundError:
        raise RuntimeError(
            "No trained model found at artifacts/model.pt. "
            "Run `python train.py` first (physics-only training takes seconds, "
            "no real data required)."
        )
    _model.eval()
    _scaler = FeatureScaler(FEATURE_RANGES)


@app.post("/predict")
def predict(payload: dict) -> PredictionResponse:
    if _model is None or _scaler is None:
        raise HTTPException(503, "Model not loaded.")

    try:
        req = PredictionRequest(**payload)
    except Exception:
        try:
            req = _case_insensitive_parse(payload)
        except Exception as e:
            raise HTTPException(422, f"Invalid prediction request: {e}")

    batch = {
        "flow_cfm": torch.tensor([req.flow_rate_cfm]),
        "inlet_line_size_in": torch.tensor([req.inlet_line_size_in]),
        "temp_c": torch.tensor([req.operating_temp_c]),
        "press_kpa": torch.tensor([req.operating_press_kpa]),
        "particle_size_micron": torch.tensor([req.particle_size_micron]),
        "particle_density_kgm3": torch.tensor([req.particle_density_kgm3]),
        "effective_turns": torch.tensor([req.effective_turns]),
        "inlet_height_ratio": torch.tensor([req.inlet_height_ratio]),
        "inlet_width_ratio": torch.tensor([req.inlet_width_ratio]),
        "outlet_diam_ratio": torch.tensor([req.outlet_diam_ratio]),
        "gas_onehot": torch.tensor([gas_type_to_onehot(req.gas_type)]),
    }

    with torch.no_grad():
        out = _model(batch, _scaler)

    return PredictionResponse(
        predicted_efficiency=float(out["efficiency_pred"].item() * 100.0),  # C# side works in %
        predicted_pressure_drop_pa=float(out["pressure_drop_pred"].item()),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


# ─────────────────────────────────────────────────────────────────────────
# FIELD-SOLVING MODEL — async job endpoints
# ─────────────────────────────────────────────────────────────────────────

class PredictFieldStartRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    barrel_diameter_mm: float = Field(alias="BarrelDiameterMm")
    barrel_height_mm: float = Field(alias="BarrelHeightMm")
    cone_height_mm: float = Field(alias="ConeHeightMm")
    exhaust_dia_mm: float = Field(alias="ExhaustDiaMm")
    exhaust_length_mm: float = Field(alias="ExhaustLengthMm")
    bottom_outlet_mm: float = Field(alias="BottomOutletMm")

    inlet_height_mm: float = Field(alias="InletHeightMm")
    inlet_width_mm: float = Field(alias="InletWidthMm")
    flow_rate_cfm: float = Field(alias="FlowRateCFM")
    operating_temp_c: float = Field(alias="OperatingTempC", default=25.0)
    operating_press_kpa: float = Field(alias="OperatingPressKPa", default=101.325)
    gas_type: str = Field(alias="GasType", default="Air")


class PredictFieldStartResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="JobId")
    status: str = Field(alias="Status")


class FieldResultDto(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    r_m: list[float] = Field(alias="RMeters")
    z_m: list[float] = Field(alias="ZMeters")
    v_r_ms: list[float] = Field(alias="VRMs")
    v_theta_ms: list[float] = Field(alias="VThetaMs")
    v_z_ms: list[float] = Field(alias="VZMs")
    pressure_pa: list[float] = Field(alias="PressurePa")
    rho_kgm3: float = Field(alias="RhoKgm3")
    nu_m2s: float = Field(alias="NuM2s")
    v_inlet_ms: float = Field(alias="VInletMs")


class PredictFieldStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="JobId")
    status: str = Field(alias="Status")  # "running" | "completed" | "failed"
    error_message: Optional[str] = Field(alias="ErrorMessage", default=None)
    result: Optional[FieldResultDto] = Field(alias="Result", default=None)


# In-process job store — see production-limitation note in module docstring.
_field_jobs: dict[str, dict] = {}
_field_jobs_lock = threading.Lock()

# Training defaults for the on-demand job. Chosen as a starting balance
# between wait time and convergence quality — validated to run cleanly
# end-to-end, NOT yet validated at exactly these epoch counts for full
# convergence; tune based on real usage once this is live.
FIELD_JOB_EPOCHS_ADAM = 3000
FIELD_JOB_EPOCHS_LBFGS = 300


def _run_field_job(job_id: str, req: PredictFieldStartRequest) -> None:
    try:
        geometry = geometry_from_dimensions_mm(
            barrel_diameter_mm=req.barrel_diameter_mm,
            barrel_height_mm=req.barrel_height_mm,
            cone_height_mm=req.cone_height_mm,
            exhaust_dia_mm=req.exhaust_dia_mm,
            exhaust_length_mm=req.exhaust_length_mm,
            bottom_outlet_mm=req.bottom_outlet_mm,
        )
        gas_onehot = torch.tensor([gas_type_to_onehot(req.gas_type)])
        rho_t, nu_t = fluid_properties(
            torch.tensor([req.operating_temp_c]),
            torch.tensor([req.operating_press_kpa]),
            gas_onehot,
        )
        rho, nu = rho_t.item(), nu_t.item()
        v_inlet = inlet_velocity_ms(
            torch.tensor([req.flow_rate_cfm]),
            torch.tensor([req.inlet_height_mm * 1e-3]),
            torch.tensor([req.inlet_width_mm * 1e-3]),
        ).item()

        def on_progress(epoch, total, loss):
            with _field_jobs_lock:
                _field_jobs[job_id]["progress"] = f"{epoch}/{total}"

        model, scaler, history = train_field_model(
            geometry, rho, nu, v_inlet,
            epochs_adam=FIELD_JOB_EPOCHS_ADAM,
            epochs_lbfgs=FIELD_JOB_EPOCHS_LBFGS,
            on_progress=on_progress,
        )

        grid = evaluate_grid(model, scaler, geometry)
        result = FieldResultDto(
            r_m=grid["r_m"], z_m=grid["z_m"],
            v_r_ms=grid["v_r_ms"], v_theta_ms=grid["v_theta_ms"],
            v_z_ms=grid["v_z_ms"], pressure_pa=grid["pressure_pa"],
            rho_kgm3=rho, nu_m2s=nu, v_inlet_ms=v_inlet,
        )

        with _field_jobs_lock:
            _field_jobs[job_id]["status"] = "completed"
            _field_jobs[job_id]["result"] = result

    except Exception as e:
        with _field_jobs_lock:
            _field_jobs[job_id]["status"] = "failed"
            _field_jobs[job_id]["error_message"] = str(e)


@app.post("/predict_field/start")
def predict_field_start(payload: dict) -> PredictFieldStartResponse:
    try:
        req = PredictFieldStartRequest(**payload)
    except Exception as e:
        raise HTTPException(422, f"Invalid field prediction request: {e}")

    job_id = str(uuid.uuid4())
    with _field_jobs_lock:
        _field_jobs[job_id] = {"status": "running", "result": None, "error_message": None, "progress": "0/0"}

    thread = threading.Thread(target=_run_field_job, args=(job_id, req), daemon=True)
    thread.start()

    return PredictFieldStartResponse(job_id=job_id, status="running")


@app.get("/predict_field/status/{job_id}")
def predict_field_status(job_id: str) -> PredictFieldStatusResponse:
    with _field_jobs_lock:
        job = _field_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No such job: {job_id}")
        return PredictFieldStatusResponse(
            job_id=job_id,
            status=job["status"],
            error_message=job["error_message"],
            result=job["result"],
        )