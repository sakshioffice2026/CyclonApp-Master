"""
train_stairmand_colab.py
─────────────────────────
Run this top-to-bottom in a Google Colab notebook to train the Stairmand
High-Efficiency (HE) cyclone field model. No command-line flags — every
setting is a plain variable in the CONFIG section below.

This is deliberately NOT a new implementation of the training loop: it
imports STAIRMAND_RATIOS and train_parametric_field_model /
save_parametric_field_checkpoint straight from field_train.py, which is
still the single source of truth for both the Lapple and Stairmand
families. This script is only a convenience wrapper around it for Colab.

WHAT YOU NEED IN THE COLAB SESSION (same folder, upload all of them):
    field_train.py
    field_model.py
    field_physics.py
    field_turbulence.py
    field_boundary_conditions.py
    sanity_check.py
    export_onnx.py
    train_stairmand_colab.py   (this file)

HOW TO RUN IN COLAB:
    1. Runtime -> Change runtime type -> GPU (optional but faster; CPU
       works fine too, this network is small).
    2. Upload the 8 files above into the Colab file browser (left sidebar)
       so they sit in /content alongside each other.
    3. In a cell:  !pip install torch --quiet   (Colab usually already has
       it; this is a no-op if so)
    4. In a new cell:  !pip install onnx --quiet   (needed for the ONNX
       export step at the bottom of this script)
    5. In a new cell:  %run train_stairmand_colab.py
       (or just paste this whole file's contents into one cell and run it)
    6. When it finishes, download the two output files from the Colab
       file browser:
           cyclone_model_stairmand.pth    (checkpoint — keep for later
                                            resume/retraining)
           cyclone_model_stairmand.onnx   (this is the file to give to
                                            the .NET side)
"""
from __future__ import annotations

import torch

from field_train import (
    STAIRMAND_RATIOS,
    train_parametric_field_model,
    save_parametric_field_checkpoint,
)
from field_physics import fluid_properties
import export_onnx


# ─────────────────────────────────────────────────────────────────────────
# CONFIG — edit these plain variables, no CLI flags needed.
# ─────────────────────────────────────────────────────────────────────────

# Family of Stairmand-shaped cyclone sizes/flow rates this ONE network will
# learn to cover. Keep these matching the realistic range your app's users
# will actually design for — the network only interpolates reliably inside
# this window (see train_parametric_field_model's docstring in
# field_train.py). These defaults mirror the same window already used for
# the Lapple parametric model, so both families are comparable.
DIAMETER_MIN_MM = 150.0
DIAMETER_MAX_MM = 750.0
FLOW_MIN_CFM = 300.0
FLOW_MAX_CFM = 13000.0

# Operating fluid conditions assumed during training. If your real designs
# span a wide temperature/pressure/gas range, you'll eventually want a
# parametric input for that too — out of scope for this first Stairmand
# pass, matching how the existing Lapple parametric model also fixes these.
OPERATING_TEMP_C = 25.0
OPERATING_PRESS_KPA = 101.325
GAS_TYPE = "Air"

# Training run length/architecture. epochs=20000 matches the Lapple
# parametric default in field_train.py's CLI and is a reasonable starting
# point; watch the printed loss and increase if it hasn't flattened out by
# the end.
EPOCHS = 20000
N_INTERIOR = 2048          # PDE collocation points resampled every epoch
HIDDEN = 64
N_LAYERS = 6
LR_ADAM_START = 3e-3
LR_ADAM_END = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 0
PROGRESS_EVERY = 25

# "cuda" if you turned on a Colab GPU runtime, else "cpu".
DEVICE = "cuda"

# Saves an intermediate checkpoint every this-many epochs (in addition to
# the final save at the end) so a Colab disconnect loses at most this many
# epochs of progress, not the whole run. Set to None to disable.
CHECKPOINT_EVERY = 2000

# If you already have a partially-trained Stairmand checkpoint (e.g. from
# a previous Colab session that disconnected), set this to its path to
# continue training instead of starting over. Leave as None for a fresh run.
RESUME_FROM: str | None = None

CHECKPOINT_PATH = "cyclone_model_stairmand.pth"
ONNX_OUTPUT_PATH = "cyclone_model_stairmand.onnx"


# ─────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────

def _print_progress(epoch: int, total: int, loss: float) -> None:
    if epoch == total or epoch % (PROGRESS_EVERY * 10) == 0:
        print(f"[{epoch:>6}/{total}] loss={loss:.6e}")


def main() -> None:
    print(
        f"Training Stairmand HE field model: "
        f"diameter=[{DIAMETER_MIN_MM}, {DIAMETER_MAX_MM}]mm, "
        f"flow=[{FLOW_MIN_CFM}, {FLOW_MAX_CFM}]CFM, gas={GAS_TYPE}, "
        f"epochs={EPOCHS}, device={DEVICE}"
    )
    print(f"Ratios (STAIRMAND_RATIOS from field_train.py): {STAIRMAND_RATIOS}")

    model, scaler, history = train_parametric_field_model(
        rho_fn=fluid_properties,
        ratios=STAIRMAND_RATIOS,
        diameter_range_m=(DIAMETER_MIN_MM * 1e-3, DIAMETER_MAX_MM * 1e-3),
        flow_rate_range_cfm=(FLOW_MIN_CFM, FLOW_MAX_CFM),
        operating_temp_c=OPERATING_TEMP_C,
        operating_press_kpa=OPERATING_PRESS_KPA,
        gas_type=GAS_TYPE,
        epochs=EPOCHS,
        n_interior=N_INTERIOR,
        hidden=HIDDEN,
        n_layers=N_LAYERS,
        lr_start=LR_ADAM_START,
        lr_end=LR_ADAM_END,
        grad_clip_norm=GRAD_CLIP_NORM,
        device=DEVICE,
        seed=SEED,
        on_progress=_print_progress,
        progress_every=PROGRESS_EVERY,
        resume_from=RESUME_FROM,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print("\n── Training done ────────────────────────────────────────")
    print(f"final_loss={history['final_loss']:.6e}  wall_time_s={history['wall_time_s']:.1f}")

    # Final authoritative save (checkpoint_every above already saved
    # intermediate copies during the run — this overwrites with the
    # actually-final weights).
    save_parametric_field_checkpoint(
        CHECKPOINT_PATH,
        model=model,
        scaler=scaler,
        ratios=STAIRMAND_RATIOS,
        operating_temp_c=OPERATING_TEMP_C,
        operating_press_kpa=OPERATING_PRESS_KPA,
        gas_type=GAS_TYPE,
        hidden=history["hidden"],
        n_layers=history["n_layers"],
    )
    print(f"Saved checkpoint -> {CHECKPOINT_PATH}")

    # ── Export straight to ONNX in the same run — calls export_onnx.py's
    # own export() function directly (no subprocess/CLI invocation), so
    # there's still only one place (export_onnx.py) that knows how to
    # build the ONNX graph.
    export_onnx.export(CHECKPOINT_PATH, ONNX_OUTPUT_PATH)
    print(f"Saved ONNX model -> {ONNX_OUTPUT_PATH}")

    print(
        "\nDone. Download both files from the Colab file browser:\n"
        f"  {CHECKPOINT_PATH}   (keep for future resume/retraining)\n"
        f"  {ONNX_OUTPUT_PATH}  (give this one to the .NET side — it goes "
        f"in Web/Models/ and gets wired up via "
        f"CyclonePredictionService:OnnxModelPathsByType -> STAIRMAND in "
        f"appsettings.json)"
    )


if __name__ == "__main__":
    main()