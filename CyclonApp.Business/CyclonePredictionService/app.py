"""
app.py
──────
FastAPI wrapper around the trained CyclonePINN. Matches the exact contract
already implemented on the .NET side in CyclonePredictionRepository.cs:

    POST {base_url}/predict
    body: PredictionRequest  (FlowRateCFM, InletLineSizeIn, OperatingTempC,
                               OperatingPressKPa, GasType, ParticleSizeMicron,
                               ParticleDensityKgm3, EffectiveTurns,
                               InletHeightRatio, InletWidthRatio, OutletDiamRatio)
    response: PredictionResponse (PredictedEfficiency, PredictedPressureDropPa)

C#'s HttpClient.PostAsJsonAsync/ReadFromJsonAsync are called without custom
JsonSerializerOptions in CyclonePredictionRepository.cs, so System.Text.Json
uses its *default* options: exact, case-sensitive PascalCase property names,
no camelCase conversion. The Pydantic models below mirror that exactly via
field aliases, and the request parser additionally falls back to a
case-insensitive match so this still works if that ever changes on the C#
side (e.g. someone adds JsonSerializerDefaults.Web later).

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from model import CyclonePINN, FeatureScaler, RAW_FEATURES
from dataset import FEATURE_RANGES
from physics import gas_type_to_onehot

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
