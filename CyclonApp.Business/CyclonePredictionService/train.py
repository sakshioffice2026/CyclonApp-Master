python -c "import torch; print(torch.__version__)""""
train.py
────────
Loss = w_reg  * ||correction||^2                 (defer to physics absent evidence)
     + w_data * MSE(prediction, measurement)      (only on rows with real labels)
     + w_mono * monotonicity_violation             (physics-informed residual)

The monotonicity term is the actual "physics-informed" part of the PINN:
using autograd on the collocation points, we require

    d(efficiency_pred)     / d(particle_size_micron) >= 0   (bigger particles
                                                               are never harder
                                                               to collect)
    d(efficiency_pred)     / d(effective_turns)       >= 0   (more turns never
                                                               hurts efficiency)
    d(pressure_drop_pred)  / d(flow_cfm)              >= 0   (more flow never
                                                               reduces pressure
                                                               drop)

These are well-established, non-controversial monotonic relationships in
cyclone separator behavior, and enforcing them via a gradient penalty is
exactly the mechanism a PINN uses to keep predictions physically sane in
regions with little or no training data — not just where the physics
baseline itself already guarantees it (it does, but the correction head
could otherwise learn to violate it; this term stops that).

Usage:
    python train.py                          # physics-only (no real data yet)
    python train.py --data real_records.csv  # blend in measured records
"""
from __future__ import annotations
import argparse
import json
import torch

from dataset import sample_synthetic_batch, load_real_csv, FEATURE_RANGES
from model import CyclonePINN, FeatureScaler


def monotonicity_penalty(model, scaler, batch, device):
    """Recompute predictions with grad-tracked inputs, penalize wrong-sign derivatives."""
    tracked = {}
    for k, v in batch.items():
        if k == "gas_onehot":
            tracked[k] = v
        else:
            tracked[k] = v.clone().detach().requires_grad_(True)

    out = model(tracked, scaler)
    eff = out["efficiency_pred"]
    dp = out["pressure_drop_pred"]

    grad_eff_dp_size = torch.autograd.grad(
        eff, tracked["particle_size_micron"], grad_outputs=torch.ones_like(eff),
        create_graph=True, retain_graph=True)[0]
    grad_eff_turns = torch.autograd.grad(
        eff, tracked["effective_turns"], grad_outputs=torch.ones_like(eff),
        create_graph=True, retain_graph=True)[0]
    grad_dp_flow = torch.autograd.grad(
        dp, tracked["flow_cfm"], grad_outputs=torch.ones_like(dp),
        create_graph=True, retain_graph=True)[0]

    # Penalize only the "wrong sign" part (negative where it must be >= 0).
    viol = (torch.relu(-grad_eff_dp_size) ** 2
            + torch.relu(-grad_eff_turns) ** 2
            + torch.relu(-grad_dp_flow) ** 2)
    return viol.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                     help="Optional CSV of real measured records (see dataset.py docstring for schema).")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-reg", type=float, default=1.0)
    ap.add_argument("--w-data", type=float, default=25.0)
    ap.add_argument("--w-mono", type=float, default=5.0)
    ap.add_argument("--out", type=str, default="artifacts")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cpu"

    model = CyclonePINN().to(device)
    scaler = FeatureScaler(FEATURE_RANGES)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    real = load_real_csv(args.data) if args.data else None
    if real is not None:
        print(f"Loaded {len(real['flow_cfm'])} real record(s) from {args.data}")
    else:
        print("No --data supplied: training physics-only. "
              "Correction heads will regularize toward ~0, i.e. this service "
              "will reproduce the deterministic Lapple/Shepherd-Lapple "
              "calculation almost exactly until real records are added.")

    for epoch in range(args.epochs):
        batch = sample_synthetic_batch(args.batch)
        out = model(batch, scaler)

        reg_loss = (out["correction_eff"] ** 2).mean() + (out["correction_dp_frac"] ** 2).mean()
        mono_loss = monotonicity_penalty(model, scaler, batch, device)

        data_loss = torch.tensor(0.0)
        if real is not None:
            real_out = model(real, scaler)
            eff_mask = ~torch.isnan(real["measured_efficiency"])
            dp_mask = ~torch.isnan(real["measured_pressure_drop_pa"])
            if eff_mask.any():
                data_loss = data_loss + torch.nn.functional.mse_loss(
                    real_out["efficiency_pred"][eff_mask],
                    real["measured_efficiency"][eff_mask])
            if dp_mask.any():
                # normalize pressure-drop error to a comparable scale to efficiency error
                dp_err = (real_out["pressure_drop_pred"][dp_mask]
                          - real["measured_pressure_drop_pa"][dp_mask]) / 500.0
                data_loss = data_loss + (dp_err ** 2).mean()

        loss = args.w_reg * reg_loss + args.w_mono * mono_loss + args.w_data * data_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  loss={loss.item():.5f}  "
                  f"reg={reg_loss.item():.5f}  mono={mono_loss.item():.5f}  "
                  f"data={data_loss.item():.5f}")

    import os
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    with open(f"{args.out}/scaler.json", "w") as f:
        json.dump(scaler.state_dict(), f)
    print(f"Saved model + scaler to {args.out}/")


if __name__ == "__main__":
    main()
