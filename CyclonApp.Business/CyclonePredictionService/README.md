# Cyclone Prediction Service — Physics-Informed Neural Network

This is the `/predict` service that `CyclonePredictionRepository.cs` already
calls (`CyclonePredictionService:BaseUrl`, default `http://localhost:8000`).
No changes are required on the .NET side — the request/response JSON shapes
match `PredictionRequest` / `PredictionResponse` exactly, including casing.

## What "physics-informed" means here, precisely

The network never predicts efficiency or pressure drop from a blank slate.
It predicts a small, bounded **correction** on top of the same
Lapple / Shepherd-Lapple / Sutherland relationships your C# calculation
engine (`CyclonCalculationRepository.cs`) already uses — reimplemented in
`physics.py` so the two stay in lockstep:

```
efficiency_pred    = clamp(physics_efficiency + correction_eff, 0, 1)
pressure_drop_pred = physics_pressure_drop * (1 + correction_dp_frac)
```

`correction_eff` is capped at ±8 percentage points, `correction_dp_frac` at
±15% — the network structurally cannot diverge far from the trusted
calculation. That matters because your existing trusted-range check in
`CyclonePredictionRepository.PredictAsync` compares the two and flags
results outside 8% as "indicative only" — this design guarantees that
check stays meaningful rather than becoming a rubber stamp.

During training, two things are enforced:

1. **Regularization toward zero correction**, so with no real data the
   service faithfully reproduces the deterministic calculation (see
   "Current behavior" below).
2. **A physics-residual (monotonicity) loss**, computed via autograd on
   synthetic collocation points — the actual PINN mechanism. It penalizes
   the network if it ever learns a correction that would make efficiency
   *decrease* as particle size or turn count increases, or make pressure
   drop *decrease* as flow increases. These are non-controversial physical
   facts about cyclone separators, enforced as differentiable constraints
   rather than left to hope.

## Current behavior (be honest about this with anyone evaluating the product)

No real commissioning/test-rig data has been added yet. Out of the box,
this service reproduces the deterministic Lapple/Shepherd-Lapple result to
within a fraction of a percent (verified: for a sample input the physics
baseline gave 28.34% efficiency / 3269.96 Pa and the trained network gave
28.35% / 3269.98 Pa). That's correct, expected behavior, not a bug — there
is currently no evidence to justify deviating from the known formula, so
the network doesn't invent any. Its value increases as real records are
added (see below).

## Files

| File | Purpose |
|---|---|
| `physics.py` | Differentiable (torch) port of the C# Lapple/Shepherd-Lapple/Sutherland/ideal-gas relationships. |
| `dataset.py` | Synthetic collocation sampling across realistic operating ranges + optional real-CSV loader. |
| `model.py` | The bounded-correction PINN architecture + feature scaler. |
| `train.py` | Training loop: regularization + monotonicity residual + optional real-data loss. |
| `app.py` | FastAPI service exposing `POST /predict` matching the C# contract exactly. |

## Running it

```bash
pip install -r requirements.txt
python train.py                 # seconds; produces artifacts/model.pt + scaler.json
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then `CyclonePredictionRepository.cs` works against it unchanged, as long as
`CyclonePredictionService:BaseUrl` in `appsettings.json` points at this
service (`http://localhost:8000` by default, matching the existing fallback
in the repository).

## Adding real data later

Once you have measured efficiency/pressure-drop values from commissioned
units or test-rig runs, put them in a CSV with this header:

```
FlowRateCFM,InletLineSizeIn,OperatingTempC,OperatingPressKPa,GasType,
ParticleSizeMicron,ParticleDensityKgm3,EffectiveTurns,InletHeightRatio,
InletWidthRatio,OutletDiamRatio,MeasuredEfficiencyPercent,MeasuredPressureDropPa
```

(Either measured column can be blank per row if only one was recorded.)

```bash
python train.py --data real_records.csv
```

The correction heads will start learning real deviations from the physics
baseline — still bounded, still cross-checked by the .NET trusted-range
logic. Re-run this whenever the dataset grows; there's no online/live
training in the service itself, by design (a served model should be a
fixed, reviewable artifact, not something that changes under a user's feet
between two identical requests).

## Retraining safety checks worth doing before shipping a new `artifacts/model.pt`

- Re-run the smoke test: predictions on a handful of known inputs should
  stay close to the physics baseline unless you have real data supporting
  a bigger correction.
- Watch the `mono` loss term in training output — it should be at or near
  zero by the end of training. If it isn't converging to ~0, increase
  `--w-mono` or reduce `--w-data`.
- Efficiency predictions must stay in [0, 100]% and pressure drop must
  stay positive — both are structurally guaranteed by the model (clamp /
  physics-baseline-times-positive-factor), not something to re-verify
  case by case, but worth a spot check after any architecture change.
