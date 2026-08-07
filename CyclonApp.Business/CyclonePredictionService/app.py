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
    body: PredictFieldStartRequest (geometry in mm + process conditions
          + CycloneTypeCode, e.g. "LAPPLE"/"STAIRMAND" — selects which
          trained checkpoint answers the request)
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
import json
import math
import threading
import time
import traceback
import uuid
from typing import Optional

import os

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from field_model import evaluate_grid, CycloneFieldPINN, FieldScaler
from field_train import run_field_prediction_job, load_parametric_field_checkpoint
from field_physics import (
    geometry_from_dimensions_mm,
    fluid_properties,
    gas_type_to_onehot,
    inlet_velocity_ms,
    inlet_axial_velocity_ms,
)
from field_turbulence import hydraulic_diameter_rect_m, inlet_turbulence_quantities
from sanity_check import mass_conservation_metrics, compute_pressure_drop
from render_field import render_cyclone_field
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Cyclone Prediction Service (Physics-Informed)")

# Where per-job CFD PNGs land, and the URL prefix they're served under.
# One subfolder per job_id so concurrent jobs (MAX_CONCURRENT_FIELD_JOBS)
# never clobber each other's cfd_result.png, and so a stale/expired job's
# image is easy to identify and sweep alongside its job-store entry.
RENDERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders")
os.makedirs(RENDERS_DIR, exist_ok=True)
app.mount("/renders", StaticFiles(directory=RENDERS_DIR), name="renders")
CAD_EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad-exports")
os.makedirs(CAD_EXPORTS_DIR, exist_ok=True)
app.mount("/cad-exports", StaticFiles(directory=CAD_EXPORTS_DIR), name="cad-exports")

# Job lifecycle / resource-limit config — see module docstring above.
MAX_CONCURRENT_FIELD_JOBS = 2
FIELD_JOB_TTL_SECONDS = 3600  # finished jobs are swept 1 hour after completion
FIELD_JOB_SWEEP_INTERVAL_SECONDS = 300

# ─────────────────────────────────────────────────────────────────────────
# PRODUCTION INFERENCE MODE — train once per cyclone family (e.g. in Colab
# via field_train.train_parametric_field_model, saved the same way
# save_parametric_field_checkpoint does), deploy each family's checkpoint
# alongside this service, and serve every request from the already-trained
# weights for the type it asks for. No training happens after deploy.
#
# ROOT-CAUSE FIX (this revision): previously this service loaded exactly
# ONE checkpoint at startup (FIELD_MODEL_CHECKPOINT_PATH, a single global
# path) and used it for every request regardless of which cyclone type the
# request was actually for — a Stairmand request would silently be
# answered by the LAPPLE-trained network (or vice versa), which is pure
# extrapolation onto a completely different family's geometry ratios, not
# a validated prediction. That's why every request could come back with
# the same "mass conservation failed" result: the model being queried
# genuinely didn't know that shape.
#
# Fixed below: requests now carry CycloneTypeCode (e.g. "LAPPLE",
# "STAIRMAND"), and this service keeps one loaded checkpoint per type,
# loaded lazily on first request and cached after that — the same
# resolve-by-type-code-then-cache pattern
# CycloneFieldOnnxPredictorProvider.cs already uses on the C# ONNX-preview
# side, so both inference paths pick the model the same way.
# ─────────────────────────────────────────────────────────────────────────
_DEFAULT_CHECKPOINT_PATHS_BY_TYPE = {
"LAPPLE": "cyclone_model_lapple_parametric.pth",
      "STAIRMAND": "cyclone_model_stairmand.pth",
    "STAIRMAND_GP": "cyclone_model_stairmand_gp.pth",
    "SWIFT_HE": "cyclone_model_swift_he.pth",
}

def _load_checkpoint_paths_by_type() -> dict[str, str]:
    """FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE, if set, must be a JSON object
    string, e.g. '{"LAPPLE": "cyclone_model.pth", "STAIRMAND": "cyclone_model_stairmand.pth"}'.
    Falls back to _DEFAULT_CHECKPOINT_PATHS_BY_TYPE (and prints a warning)
    if the env var is unset, empty, or not valid JSON, so a misconfigured
    deployment still starts up rather than crashing on the first request."""
    raw = os.environ.get("FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE")
    if not raw:
        return dict(_DEFAULT_CHECKPOINT_PATHS_BY_TYPE)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("must be a non-empty JSON object")
        return {str(k).upper(): str(v) for k, v in parsed.items()}
    except Exception as e:
        print(f"[startup] WARNING: FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE is set "
              f"but could not be parsed ({e}) — falling back to the built-in "
              f"default mapping: {_DEFAULT_CHECKPOINT_PATHS_BY_TYPE}")
        return dict(_DEFAULT_CHECKPOINT_PATHS_BY_TYPE)


FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE = _load_checkpoint_paths_by_type()

# One entry per cyclone type, populated lazily on first request for that
# type (see _get_inference_state) and reused after that — mirrors
# CycloneFieldOnnxPredictorProvider's ConcurrentDictionary cache.
_inference_states: dict[str, dict] = {}
_inference_states_lock = threading.Lock()


def _load_field_checkpoint(path: str) -> dict:
    """Loads a checkpoint saved by field_train.save_parametric_field_checkpoint
    (--mode parametric). Only model weights + the parametric FieldScaler
    (D_min/D_max/Q_min/Q_max normalization window) are needed at serve time —
    _run_field_job recomputes everything else (geometry, rho, nu, v_inlet,
    turbulence quantities) per-request from the incoming
    PredictFieldStartRequest, since those vary by request now (see
    PRODUCTION INFERENCE MODE note above). Delegates to
    field_train.load_parametric_field_checkpoint so app.py and
    field_train.py's --resume-from share one implementation of the
    checkpoint file format.

    FALLBACK: if the checkpoint on disk turns out to be a --mode single
    (save_field_checkpoint) checkpoint instead — no "checkpoint_kind" key,
    just model_state_dict/hidden/n_layers/scaler_state_dict plus the
    fixed-geometry training constants — load it directly rather than
    refusing to start. The network still takes diameter_m/flow_rate_cfm as
    explicit inputs either way (see field_model.py), so it will still
    produce an output for whatever the request asks for; it just means
    this particular checkpoint was only ever trained at one geometry/flow
    point, so requests far from that point are extrapolating, not
    interpolating a learned family the way a true parametric checkpoint
    would. Good enough to unblock serving; swap in a real --mode parametric
    checkpoint when one exists.
    """
    try:
        loaded = load_parametric_field_checkpoint(path)
        return {"model": loaded["model"], "scaler": loaded["scaler"]}
    except RuntimeError:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in raw or "scaler_state_dict" not in raw:
            raise  # genuinely not a checkpoint we know how to read at all

        print(f"[startup] WARNING: '{path}' is a single-geometry checkpoint "
              f"(--mode single), not a parametric one. Loading it anyway as "
              f"a fallback so the service can start — predictions away from "
              f"this checkpoint's original training geometry/flow rate will "
              f"extrapolate. Retrain with --mode parametric when possible.")

        model = CycloneFieldPINN(hidden=raw["hidden"], n_layers=raw["n_layers"])
        model.load_state_dict(raw["model_state_dict"])
        model.eval()
        scaler = FieldScaler.from_state_dict(raw["scaler_state_dict"])
        return {"model": model, "scaler": scaler}


def _get_inference_state(cyclone_type_code: str) -> dict:
    """Resolves the loaded {"model", "scaler"} state for a cyclone type
    code, loading and caching it on first use. Falls back to LAPPLE (the
    first entry in FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE) with a loud
    warning if the requested type has no configured checkpoint path yet,
    rather than silently answering with whatever type happened to load
    first — that silent-fallback behavior is exactly the bug this fix
    replaces."""
    key = (cyclone_type_code or "").strip().upper()

    with _inference_states_lock:
        if key in _inference_states:
            return _inference_states[key]

        path = FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE.get(key)
        if path is None:
            fallback_key = next(iter(FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE))
            print(f"[predict_field] WARNING: no checkpoint configured for "
                  f"cyclone type '{cyclone_type_code}' — falling back to "
                  f"'{fallback_key}'. Predictions will be for the wrong "
                  f"cyclone family's shape. Add an entry to "
                  f"FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE once this type has "
                  f"been trained.")
            if fallback_key in _inference_states:
                return _inference_states[fallback_key]
            path = FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE[fallback_key]
            key = fallback_key

        state = _load_field_checkpoint(path)
        _inference_states[key] = state
        print(f"[predict_field] Loaded inference-only field model for "
              f"'{key}' from {path}")
        return state


@app.on_event("startup")
def load_field_model_checkpoints():
    """Eagerly loads every configured type's checkpoint at startup (rather
    than waiting for the first request) so a missing/corrupt file is
    caught immediately in the logs, not on some later request. A type
    failing to load here does not stop the others or crash the service —
    it's just unavailable until fixed, same as if it had never been
    configured."""
    for key, path in FIELD_MODEL_CHECKPOINT_PATHS_BY_TYPE.items():
        try:
            _get_inference_state(key)
        except Exception as e:
            print(f"[startup] WARNING: failed to load checkpoint for "
                  f"'{key}' from '{path}': {e}. Requests for this cyclone "
                  f"type will fail (or fall back) until this is fixed.")


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

    # Selects which trained checkpoint answers this request — see
    # PRODUCTION INFERENCE MODE note above. Defaults to "LAPPLE" only for
    # backward compatibility with any caller that predates this field;
    # every current caller (CyclonePredictionRepository.StartFieldPredictionAsync)
    # sends the design's actual CycloneType.Code explicitly.
    cyclone_type_code: str = Field(alias="CycloneTypeCode", default="LAPPLE")


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

    # Inlet-ring-average minus vortex-finder-bore-average static pressure
    # at the z=0 plane (see compute_pressure_drop in sanity_check.py) --
    # the field-solve's own estimate of the SAME quantity the
    # Shepherd-Lapple baseline's dP_Pa describes. Optional/nullable: None
    # means compute_pressure_drop couldn't find usable points on both
    # sides of r_exhaust (e.g. too coarse a grid), NOT that the drop is
    # zero. Deliberately separate from the existing "pressure_pa" field
    # (the raw per-point field, still needed for visualization) so this
    # doesn't change that field's meaning for any existing consumer.
    pressure_drop_pa: Optional[float] = Field(alias="PressureDropPa", default=None)

    # Mass-conservation diagnostics — optional/nullable to match the .NET
    # side's FieldResultDto, which treats their absence as "not computed"
    # rather than an error.
    mass_conservation_status: Optional[str] = Field(alias="MassConservationStatus", default=None)
    mass_flow_spread: Optional[float] = Field(alias="MassFlowSpread", default=None)
    final_loss: Optional[float] = Field(alias="FinalLoss", default=None)

    # Relative URL (e.g. "/renders/<job_id>/cfd_result.png") for the
    # matplotlib CFD-style contour PNG produced by render_field.py's
    # render_cyclone_field, mounted via the /renders StaticFiles app
    # above. None means rendering hasn't run yet or failed — see
    # _run_field_job's render step, which never lets a rendering failure
    # fail the underlying field-solve job itself.
    png_url: Optional[str] = Field(alias="PngUrl", default=None)

    # Short exception message (e.g. "ValueError: ...") when png_url is
    # None because rendering failed -- the full traceback is written to
    # render_error.log next to where the PNG would have landed (see
    # RENDERS_DIR/<job_id>/), reachable at /renders/<job_id>/render_error.log.
    # This field is just enough for the client UI to show *something*
    # concrete instead of a generic "could not be rendered" message.
    render_error: Optional[str] = Field(alias="RenderError", default=None)


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


def _warn_if_request_outside_trained_range(req: PredictFieldStartRequest, state: dict) -> None:
    """The parametric checkpoint was trained across a diameter/flow-rate
    window (state["scaler"].D_min/D_max/Q_min/Q_max — see
    train_parametric_field_model). A request inside that window is
    interpolation, which the network was trained for; a request outside it
    is extrapolation, which it was not — this does not block or alter the
    request (429/422 are for auth/format errors, not this), it only makes
    an out-of-range design visible in logs rather than silently trusting an
    unvalidated network output."""
    scaler = state["scaler"]
    diameter_m = req.barrel_diameter_mm * 1e-3
    if not (scaler.D_min <= diameter_m <= scaler.D_max):
        print(
            f"[predict_field] WARNING: request barrel_diameter_mm="
            f"{req.barrel_diameter_mm} ({diameter_m} m) is outside the "
            f"trained range [{scaler.D_min}, {scaler.D_max}] m — result is "
            f"extrapolation, not validated interpolation."
        )
    if not (scaler.Q_min <= req.flow_rate_cfm <= scaler.Q_max):
        print(
            f"[predict_field] WARNING: request flow_rate_cfm="
            f"{req.flow_rate_cfm} is outside the trained range "
            f"[{scaler.Q_min}, {scaler.Q_max}] CFM — result is "
            f"extrapolation, not validated interpolation."
        )


def _run_field_job(job_id: str, req: PredictFieldStartRequest) -> None:
    global _field_jobs_running_count
    try:
        # Resolves to the checkpoint trained for THIS request's cyclone
        # family (see PRODUCTION INFERENCE MODE note above) — no longer a
        # single global state used for every type.
        state = _get_inference_state(req.cyclone_type_code)

        _warn_if_request_outside_trained_range(req, state)

        # INFERENCE ONLY — no training here. state["model"] supplies the
        # trained weights; everything below this line is request-specific
        # glue (same as field_train.run_field_prediction_job's geometry ->
        # fluid properties -> inlet velocity -> inlet turbulence pipeline),
        # so the network is actually queried at the design that was asked
        # for, not the checkpoint's last-trained scale.

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
        v_z_inlet = inlet_axial_velocity_ms(
            torch.tensor([req.flow_rate_cfm]),
            r_barrel_m=geometry.r_barrel, r_exhaust_m=geometry.r_exhaust,
        ).item()

        # Physical (L/U/P/K/E) scale for THIS request's geometry/velocity;
        # the parametric D/Q normalization window is carried over unchanged
        # from the trained checkpoint (see FieldScaler.with_scales — it
        # must stay fixed across every query, not just every training
        # epoch, for D_norm/Q_norm to mean the same thing they meant during
        # training).
        eval_scaler = state["scaler"].with_scales(
            length_scale_m=geometry.r_barrel,
            velocity_scale_ms=max(v_inlet, 1e-6),
            rho=rho,
        )

        diameter_m = req.barrel_diameter_mm * 1e-3
        grid = evaluate_grid(
            state["model"], eval_scaler, geometry,
            diameter_m=diameter_m,
            flow_rate_cfm=req.flow_rate_cfm,
        )

        q_design = float(v_z_inlet) * math.pi * (
            geometry.r_barrel ** 2 - geometry.r_exhaust ** 2
        )
        mc = mass_conservation_metrics(grid, q_design=q_design)
        grid["mass_conservation_status"] = mc["status"]
        grid["mass_flow_spread"] = mc["rel_spread"]

        # See compute_pressure_drop's docstring for why this -- not
        # max(pressure_pa) - min(pressure_pa) over the whole grid -- is
        # the quantity comparable to CyclonCalculationRepository.cs's
        # Shepherd-Lapple dP_Pa.
        pdrop = compute_pressure_drop(grid, r_exhaust_m=geometry.r_exhaust, rho_kgm3=rho)
        if pdrop["pressure_drop_pa"] is None:
            print(
                f"[predict_field] WARNING: could not compute inlet/outlet "
                f"pressure drop for job {job_id}: {pdrop['detail']}"
            )

        # CFD-style contour PNG (render_field.py) — same in-process,
        # no-IPC-needed rationale documented in that module's docstring.
        # A rendering failure must never fail the underlying field-solve
        # job the client is waiting on, so it's caught and logged, and
        # png_url is simply left None (client-side UI treats that as
        # "image unavailable", not an error state for the whole job).
        png_url = None
        render_error = None
        try:
            job_output_dir = os.path.join(RENDERS_DIR, job_id)
            geometry_mm = dict(
                barrel_diameter_mm=req.barrel_diameter_mm,
                barrel_height_mm=req.barrel_height_mm,
                cone_height_mm=req.cone_height_mm,
                exhaust_dia_mm=req.exhaust_dia_mm,
                exhaust_length_mm=req.exhaust_length_mm,
                bottom_outlet_mm=req.bottom_outlet_mm,
            )
            render_cyclone_field(
                grid=grid,
                geometry_mm=geometry_mm,
                output_dir=job_output_dir,
                known_efficiency_percent=None,  # not available service-side; .NET has its own value
                known_pressure_drop_pa=pdrop["pressure_drop_pa"],
            )
            png_url = f"/renders/{job_id}/cfd_result.png"
        except Exception as render_exc:
            tb_text = traceback.format_exc()
            render_error = f"{type(render_exc).__name__}: {render_exc}"
            print(
                f"[predict_field] WARNING: CFD PNG render failed for job "
                f"{job_id}:\n{tb_text}"
            )
            # Persisted alongside wherever the PNG would have landed, so
            # the actual traceback is inspectable via the same /renders
            # static mount -- previously this only went to the server
            # console, which is unreachable to anyone not SSH'd into the
            # box at the moment it happened.
            try:
                os.makedirs(job_output_dir, exist_ok=True)
                with open(os.path.join(job_output_dir, "render_error.log"), "w", encoding="utf-8") as f:
                    f.write(tb_text)
            except OSError:
                pass

        field_result = FieldResultDto(
            r_m=grid["r_m"], z_m=grid["z_m"],
            v_r_ms=grid["v_r_ms"], v_theta_ms=grid["v_theta_ms"],
            v_z_ms=grid["v_z_ms"], pressure_pa=grid["pressure_pa"],
            rho_kgm3=rho, nu_m2s=nu, v_inlet_ms=v_inlet,
            pressure_drop_pa=pdrop["pressure_drop_pa"],
            # Mass-conservation diagnostics — see run_field_prediction_job's
            # "Root-cause fix" comment. Previously always omitted (None on
            # the wire), which made the .NET Engineering Insights panel
            # treat every completed job as a mass-conservation failure.
            mass_conservation_status=grid.get("mass_conservation_status"),
            mass_flow_spread=grid.get("mass_flow_spread"),
            final_loss=None,  # no training occurred for this request
            png_url=png_url,
            render_error=render_error,
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

    # Outside the lock: filesystem I/O shouldn't hold up job-store access,
    # and these directories are keyed by job_id (UUID) so there's no
    # cross-job collision risk in deleting them after the fact.
    for jid in expired:
        job_render_dir = os.path.join(RENDERS_DIR, jid)
        if os.path.isdir(job_render_dir):
            try:
                import shutil
                shutil.rmtree(job_render_dir, ignore_errors=True)
            except OSError:
                pass


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


        # ─────────────────────────────────────────────────────────────────────────
# CAD GENERATION — synchronous (seconds, not minutes; no job/poll needed,
# unlike /predict_field). Add this block to app.py, after the existing
# /predict_field/* routes. Also add near the top of app.py:
#     from cad_generator import generate_cyclone_cad
# and add this static mount alongside the existing RENDERS_DIR mount:
#     CAD_EXPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cad-exports")
#     os.makedirs(CAD_EXPORTS_DIR, exist_ok=True)
#     app.mount("/cad-exports", StaticFiles(directory=CAD_EXPORTS_DIR), name="cad-exports")
# ─────────────────────────────────────────────────────────────────────────

class GenerateCadRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # PascalCase aliases to match CyclonePredictionRepository.cs's
    # OutgoingJsonOptions (PropertyNamingPolicy = null), same convention
    # as PredictFieldStartRequest above.
    RevisionId: int = Field(..., alias="RevisionId")
    BarrelDiameterMm: float = Field(..., alias="BarrelDiameterMm")
    BarrelHeightMm: float = Field(..., alias="BarrelHeightMm")
    ConeHeightMm: float = Field(..., alias="ConeHeightMm")
    ExhaustDiaMm: float = Field(..., alias="ExhaustDiaMm")
    ExhaustLengthMm: float = Field(..., alias="ExhaustLengthMm")
    BottomOutletMm: float = Field(..., alias="BottomOutletMm")
    InletHeightMm: float = Field(..., alias="InletHeightMm")
    InletWidthMm: float = Field(..., alias="InletWidthMm")


class GenerateCadResponse(BaseModel):
    StepUrl: Optional[str] = None
    DxfUrl: Optional[str] = None
    PdfUrl: Optional[str] = None


@app.post("/generate_cad", response_model=GenerateCadResponse)
def generate_cad(request: GenerateCadRequest):
    """
    Synchronous CAD generation — builds cyclone geometry from dimensions
    and exports STEP/DXF/PDF. Unlike /predict_field/start, no job store or
    polling: FreeCAD geometry+export takes seconds, so the HTTP request
    just waits for the result directly.
    """
    print("DEBUG: /generate_cad called")
    print(f"DEBUG: request data = {request.dict()}")
    from cad_generator import generate_cyclone_cad
    print("DEBUG: cad_generator imported")

    output_dir = os.path.join(CAD_EXPORTS_DIR, str(request.RevisionId))

    dims = {
        "RevisionId": request.RevisionId,
        "BarrelDiameterMm": request.BarrelDiameterMm,
        "BarrelHeightMm": request.BarrelHeightMm,
        "ConeHeightMm": request.ConeHeightMm,
        "ExhaustDiaMm": request.ExhaustDiaMm,
        "ExhaustLengthMm": request.ExhaustLengthMm,
        "BottomOutletMm": request.BottomOutletMm,
        "InletHeightMm": request.InletHeightMm,
        "InletWidthMm": request.InletWidthMm,
    }

    try:
        result = generate_cyclone_cad(dims, output_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CAD generation failed: {e}")

    # cad_generator.generate_cyclone_cad already returns fully-formatted
    # URLs (StepUrl/DxfUrl/PdfUrl), so just pass them through.
    return GenerateCadResponse(
        StepUrl=result.get("StepUrl"),
        DxfUrl=result.get("DxfUrl"),
        PdfUrl=result.get("PdfUrl"),
    )