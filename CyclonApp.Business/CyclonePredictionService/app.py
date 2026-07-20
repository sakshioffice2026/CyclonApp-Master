"""
app.py
──────
FastAPI wrapper around async job endpoints for the field-solving
physics-guided model (velocity/pressure fields).

RETIRED: the old synchronous /predict endpoint (scalar CyclonePINN
correction model) and everything it depended on — model.py, dataset.py,
train.py, and physics.py's lapple_forward — have been deleted. The
.NET side's CyclonePredictionRepository.PredictAsync() /
DesignController.PredictWithModel still POST to /predict; until that
call site is removed too, it will fail (caught by the existing .NET
try/catch, surfaced as "prediction service unavailable") rather than
crash this service. /predict_field/* is unaffected and remains the
only prediction contract this service serves.

CONTRACT (field-solving model):
    POST {base_url}/predict_field/start
    body: PredictFieldStartRequest (geometry in mm + process conditions)
    response: PredictFieldStartResponse (JobId, Status="running")

    GET {base_url}/predict_field/status/{job_id}
    response: PredictFieldStatusResponse (JobId, Status, ErrorMessage, Result)

Why async: unlike CyclonePINN, CycloneFieldPINN is trained fresh per
geometry/operating-point (see field_train.py docstring) — a full physics
solve takes real minutes, not milliseconds, so this cannot be a synchronous
request the way /predict is. The client starts a job, polls status.

The geometry/fluid-property glue and the train+evaluate pipeline live in
field_train.run_field_prediction_job — the single shared implementation
also used by field_train.py's CLI (`python field_train.py ...`), so offline
tuning and the live service can never drift apart.

Known production limitation, stated plainly: the job store below is an
in-process dict. It does not survive a service restart and does not work
across multiple uvicorn worker processes. Fine for a single-worker
deployment; if this service is ever scaled to multiple workers, the job
store needs to move to something shared (Redis, a DB table) — not done
here because it isn't needed yet and would be an unnecessary abstraction
for the current single-worker deployment.

Job lifecycle / resource limits:
  - MAX_CONCURRENT_FIELD_JOBS caps how many training jobs can run at once.
    A request past that cap gets an immediate 429, not an unbounded queue
    of background threads silently competing for CPU.
  - FIELD_JOB_TTL_SECONDS bounds how long a finished (completed/failed)
    job's result stays in the in-process dict before a periodic sweep
    evicts it.
  - created_at / completed_at timestamps are attached to every job so the
    client can reason about job age.

C#'s HttpClient.PostAsJsonAsync/ReadFromJsonAsync are called without custom
JsonSerializerOptions in CyclonePredictionRepository.cs, so System.Text.Json
uses its *default* options: exact, case-sensitive PascalCase property names,
no camelCase conversion. All Pydantic models below mirror that via field
aliases, consistent with the /predict_field contract.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import math
import threading
import time
import uuid
from typing import Optional

import os

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from field_model import CycloneFieldPINN, FieldScaler, evaluate_grid
from field_train import run_field_prediction_job
from sanity_check import mass_conservation_metrics

app = FastAPI(title="Cyclone Prediction Service (Physics-Informed)")


# Job lifecycle / resource-limit config — see module docstring above.
MAX_CONCURRENT_FIELD_JOBS = 2
FIELD_JOB_TTL_SECONDS = 3600  # finished jobs are swept 1 hour after completion
FIELD_JOB_SWEEP_INTERVAL_SECONDS = 300

# ─────────────────────────────────────────────────────────────────────────
# PRODUCTION INFERENCE MODE — train once (e.g. in Colab via
# `python field_train.py ... --save-checkpoint cyclone_model.pth`), deploy
# the checkpoint alongside this service, load it once at startup, and serve
# every request from the already-trained weights. No training happens after
# deploy: _run_field_job below calls evaluate_grid() on the loaded model
# instead of run_field_prediction_job()'s train_field_model() call.
#
# IMPORTANT — same caveat as field_train.py's checkpoint-save comment: this
# checkpoint is trained for ONE fixed geometry/operating point (see
# field_train.py module docstring — CycloneFieldPINN bakes geometry in as
# constants, it does not take them as inputs). Every request is served from
# that single trained design; incoming geometry/flow/gas fields on the
# request are accepted (so the existing PredictFieldStartRequest/.NET
# contract is unchanged) but do not change the result. A mismatch warning
# is logged so this isn't a silent surprise if the UI is later used to vary
# designs against a checkpoint trained for a different one.
# ─────────────────────────────────────────────────────────────────────────
FIELD_MODEL_CHECKPOINT_PATH = os.environ.get(
    "FIELD_MODEL_CHECKPOINT_PATH", "cyclone_model.pth"
)

_inference_state: Optional[dict] = None  # populated by _load_field_checkpoint()


def _load_field_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CycloneFieldPINN(hidden=ckpt["hidden"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = FieldScaler.from_state_dict(ckpt["scaler_state_dict"])
    return {
        "model": model,
        "scaler": scaler,
        "geometry": ckpt["geometry"],
        "rho": ckpt["rho"],
        "nu": ckpt["nu"],
        "v_inlet": ckpt["v_inlet"],
        "v_z_inlet": ckpt["v_z_inlet"],
        "k_inlet": ckpt["k_inlet"],
        "eps_inlet": ckpt["eps_inlet"],
    }


@app.on_event("startup")
def load_field_model_checkpoint():
    global _inference_state
    _inference_state = _load_field_checkpoint(FIELD_MODEL_CHECKPOINT_PATH)
    print(f"[startup] Loaded inference-only field model from "
          f"{FIELD_MODEL_CHECKPOINT_PATH}")


@app.get("/health")
def health():
    """No scalar model to report on anymore — this just confirms the
    service process is up and able to accept field-prediction jobs."""
    return {"status": "ok"}


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

    # Mass-conservation diagnostics — optional/nullable to match the .NET
    # side's FieldResultDto, which treats their absence as "not computed"
    # rather than an error.
    mass_conservation_status: Optional[str] = Field(alias="MassConservationStatus", default=None)
    mass_flow_spread: Optional[float] = Field(alias="MassFlowSpread", default=None)
    final_loss: Optional[float] = Field(alias="FinalLoss", default=None)


class PredictFieldStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="JobId")
    status: str = Field(alias="Status")  # "running" | "completed" | "failed"
    error_message: Optional[str] = Field(alias="ErrorMessage", default=None)
    result: Optional[FieldResultDto] = Field(alias="Result", default=None)
    created_at_unix: Optional[float] = Field(alias="CreatedAtUnix", default=None)
    completed_at_unix: Optional[float] = Field(alias="CompletedAtUnix", default=None)


# In-process job store — see production-limitation note in module docstring.
_field_jobs: dict[str, dict] = {}
_field_jobs_lock = threading.Lock()
_field_jobs_running_count = 0  # guarded by _field_jobs_lock

# NOT USED by the live path anymore (see PRODUCTION INFERENCE MODE above —
# _run_field_job now serves the checkpoint loaded at startup instead of
# calling run_field_prediction_job/training). Left in place only for the
# offline retraining path (field_train.py's own CLI is the actual "how do I
# retrain" entry point going forward); harmless if unused.
FIELD_JOB_EPOCHS_ADAM = 5
FIELD_JOB_EPOCHS_LBFGS = 2


def _warn_if_request_differs_from_trained_design(req: PredictFieldStartRequest) -> None:
    """The loaded checkpoint was trained for one fixed geometry/operating
    point (see PRODUCTION INFERENCE MODE note above). This does not block
    or alter the request — it only makes a design mismatch visible in logs
    instead of silently returning results for a different design than the
    one requested."""
    geo = _inference_state["geometry"]
    trained_r_barrel_mm = geo.r_barrel * 2000.0
    if abs(trained_r_barrel_mm - req.barrel_diameter_mm) > 1e-3:
        print(
            f"[predict_field] WARNING: request barrel_diameter_mm="
            f"{req.barrel_diameter_mm} differs from the checkpoint's trained "
            f"design ({trained_r_barrel_mm} mm). Serving the trained "
            f"design's precomputed field regardless — this checkpoint does "
            f"not retrain per request."
        )


def _run_field_job(job_id: str, req: PredictFieldStartRequest) -> None:
    global _field_jobs_running_count
    try:
        _warn_if_request_differs_from_trained_design(req)

        # INFERENCE ONLY — no training here. Uses the model/scaler/geometry
        # loaded once at startup from FIELD_MODEL_CHECKPOINT_PATH.
        state = _inference_state
        grid = evaluate_grid(state["model"], state["scaler"], state["geometry"])

        q_design = float(state["v_z_inlet"]) * math.pi * (
            state["geometry"].r_barrel ** 2 - state["geometry"].r_exhaust ** 2
        )
        mc = mass_conservation_metrics(grid, q_design=q_design)
        grid["mass_conservation_status"] = mc["status"]
        grid["mass_flow_spread"] = mc["rel_spread"]

        field_result = FieldResultDto(
            r_m=grid["r_m"], z_m=grid["z_m"],
            v_r_ms=grid["v_r_ms"], v_theta_ms=grid["v_theta_ms"],
            v_z_ms=grid["v_z_ms"], pressure_pa=grid["pressure_pa"],
            rho_kgm3=state["rho"], nu_m2s=state["nu"], v_inlet_ms=state["v_inlet"],
            # Mass-conservation diagnostics — see run_field_prediction_job's
            # "Root-cause fix" comment. Previously always omitted (None on
            # the wire), which made the .NET Engineering Insights panel
            # treat every completed job as a mass-conservation failure.
            mass_conservation_status=grid.get("mass_conservation_status"),
            mass_flow_spread=grid.get("mass_flow_spread"),
            final_loss=None,  # no training occurred for this request
        )

        with _field_jobs_lock:
            if job_id in _field_jobs:
                _field_jobs[job_id]["status"] = "completed"
                _field_jobs[job_id]["result"] = field_result
                _field_jobs[job_id]["completed_at"] = time.time()

    except Exception as e:
        with _field_jobs_lock:
            if job_id in _field_jobs:
                _field_jobs[job_id]["status"] = "failed"
                _field_jobs[job_id]["error_message"] = str(e)
                _field_jobs[job_id]["completed_at"] = time.time()

    finally:
        # Always release the concurrency slot, even if the job store entry
        # was already swept out from under us (shouldn't happen given the
        # TTL is much longer than any realistic training run, but this
        # must not leak a slot if it ever does).
        with _field_jobs_lock:
            _field_jobs_running_count -= 1


def _sweep_expired_field_jobs() -> None:
    """Evicts finished jobs older than FIELD_JOB_TTL_SECONDS from the
    in-process store, so a long-running service doesn't accumulate every
    job result ever produced. Runs on a background daemon thread; see
    _start_field_job_sweeper. Must hold _field_jobs_lock for the whole
    scan-and-delete to stay consistent with concurrent job starts/updates."""
    now = time.time()
    with _field_jobs_lock:
        expired = [
            jid for jid, job in _field_jobs.items()
            if job["status"] in ("completed", "failed")
            and job.get("completed_at") is not None
            and (now - job["completed_at"]) > FIELD_JOB_TTL_SECONDS
        ]
        for jid in expired:
            del _field_jobs[jid]


def _field_job_sweeper_loop() -> None:
    while True:
        time.sleep(FIELD_JOB_SWEEP_INTERVAL_SECONDS)
        try:
            _sweep_expired_field_jobs()
        except Exception:
            # A sweep failure must never take down the service or stop
            # future sweeps — worst case is the store grows until the next
            # successful sweep, which is exactly the pre-TTL behavior.
            pass


def _start_field_job_sweeper() -> None:
    threading.Thread(target=_field_job_sweeper_loop, daemon=True).start()


@app.on_event("startup")
def start_field_job_sweeper():
    _start_field_job_sweeper()


@app.post("/predict_field/start")
def predict_field_start(payload: dict) -> PredictFieldStartResponse:
    global _field_jobs_running_count

    try:
        req = PredictFieldStartRequest(**payload)
    except Exception as e:
        raise HTTPException(422, f"Invalid field prediction request: {e}")

    with _field_jobs_lock:
        if _field_jobs_running_count >= MAX_CONCURRENT_FIELD_JOBS:
            raise HTTPException(
                429,
                f"Too many field-prediction jobs running "
                f"(max {MAX_CONCURRENT_FIELD_JOBS}). Try again shortly.",
            )
        job_id = str(uuid.uuid4())
        _field_jobs[job_id] = {
            "status": "running",
            "result": None,
            "error_message": None,
            "progress": "0/0",
            "created_at": time.time(),
            "completed_at": None,
        }
        _field_jobs_running_count += 1

    thread = threading.Thread(target=_run_field_job, args=(job_id, req), daemon=True)
    thread.start()

    return PredictFieldStartResponse(job_id=job_id, status="running")


@app.get("/predict_field/status/{job_id}")
def predict_field_status(job_id: str) -> PredictFieldStatusResponse:
    with _field_jobs_lock:
        job = _field_jobs.get(job_id)
        if job is None:
            raise HTTPException(
                404,
                f"No such job: {job_id}. It may never have existed, or it "
                f"finished more than {FIELD_JOB_TTL_SECONDS}s ago and was "
                f"cleaned up.",
            )
        return PredictFieldStatusResponse(
            job_id=job_id,
            status=job["status"],
            error_message=job["error_message"],
            result=job["result"],
            created_at_unix=job.get("created_at"),
            completed_at_unix=job.get("completed_at"),
        )