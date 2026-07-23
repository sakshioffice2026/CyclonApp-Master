import re

BASE = "/content/CyclonApp-Master/CyclonApp.Business/CyclonePredictionService"

# ── 1. Patch sanity_check.py: replace the ENTIRE compute_pressure_drop
#      function (whatever state it's currently in) with the final,
#      known-good version -- located by function boundary, not by
#      guessing which intermediate variant is on disk. ─────────────────
sc_path = f"{BASE}/sanity_check.py"
with open(sc_path, "r", encoding="utf-8") as f:
    sc = f.read()

NEW_FUNC = '''def compute_pressure_drop(
\tgrid: dict, r_exhaust_m: float, rho_kgm3: float | None = None,
) -> dict:
\t"""Pressure drop between the inlet ring and the vortex-finder gas
\toutlet -- both of which sit at the SAME z=0 plane in this axisymmetric
\tmodel (see CycloneAxisymGeometry.sample_inlet_ring /
\tsample_top_exhaust_outlet in field_physics.py), distinguished only by
\tradius, not by height: r in [r_exhaust, r_barrel] is the inlet ring,
\tr in [0, r_exhaust) is the vortex-finder bore (gas outlet).

\tThis is the quantity the Shepherd-Lapple baseline's
\tdP_Pa = Nh * 0.5 * rho * v_inlet**2 (CyclonCalculationRepository.cs)
\tis actually describing: total-pressure loss along the flow path,
\tinlet -> outlet. It is NOT the same thing as naively taking
\tmax(pressure_pa) - min(pressure_pa) over the whole grid, which mostly
\tmeasures the radial (centrifugal) pressure spread -- high at the
\touter wall, low in the vortex core -- rather than the axial
\tinlet-to-outlet loss.

\tPRESSURE CONVENTION (confirmed via debug_pressure_breakdown.py against
\ta real trained checkpoint, D=400mm/Q=3000cfm case): the vortex-finder
\texit at z=0 carries a large axial exit velocity (~50 m/s measured,
\tvs. ~15 m/s smeared inlet axial velocity -- by continuity, the same
\tflow that entered spread over the full inlet ring must squeeze
\tthrough the much smaller exhaust bore). If that exit kinetic energy
\tis counted as still "retained" (i.e. total pressure at the outlet
\ttoo), the field pressure drop comes out far below the Shepherd-Lapple
\tbaseline, because Shepherd-Lapple's Nh was empirically calibrated
\tfrom real pressure-tap measurements taken where the exit jet's
\tkinetic energy has already dissipated -- i.e. real installations
\tcount that exit velocity head as a LOSS, not recovered pressure.
\tSo: inlet uses TOTAL pressure (static + dynamic -- the full head
\tdriving flow into the system), outlet uses STATIC pressure ONLY
\t(the exit dynamic head is treated as dissipated/lost, matching how
\tthe baseline was actually measured), when rho_kgm3 is provided.

\tReturns:
\t\t{"inlet_pressure_pa": float | None,
\t\t "outlet_pressure_pa": float | None,
\t\t "pressure_drop_pa": float | None,
\t\t "detail": str}
\t\tAll three values are None if there's no usable z~0 plane data
\t\t(e.g. an empty/degenerate grid, or one side of r_exhaust has no
\t\tsampled points) -- callers should treat that as "not computed",
\t\tnever silently substitute a zero pressure drop.
\t"""
\tgroups = _group_by_z(grid)
\tif not groups:
\t\treturn {
\t\t\t"inlet_pressure_pa": None, "outlet_pressure_pa": None,
\t\t\t"pressure_drop_pa": None, "detail": "empty grid",
\t\t}

\tz0 = min(groups.keys())  # top plane, z=0 (or nearest sampled value to it)
\tpts = groups[z0]  # list of (r, vr, vt, vz, p)

\tdef _total_p(vr: float, vt: float, vz: float, p: float) -> float:
\t\tif rho_kgm3 is None:
\t\t\treturn p
\t\treturn p + 0.5 * rho_kgm3 * (vr * vr + vt * vt + vz * vz)

\t# Inlet: total pressure (full driving head). Outlet: static only --
\t# the exit velocity head is a dissipated loss, not retained pressure
\t# (see docstring above). rho_kgm3=None keeps both static-only, for
\t# backward compatibility with any caller not passing rho.
\tinlet_p = [_total_p(vr, vt, vz, p) for (r, vr, vt, vz, p) in pts if r >= r_exhaust_m]
\toutlet_p = [p for (r, vr, vt, vz, p) in pts if r < r_exhaust_m]

\tif not inlet_p or not outlet_p:
\t\treturn {
\t\t\t"inlet_pressure_pa": None, "outlet_pressure_pa": None,
\t\t\t"pressure_drop_pa": None,
\t\t\t"detail": (
\t\t\t\tf"z={z0} plane had no points on one side of "
\t\t\t\tf"r_exhaust_m={r_exhaust_m} ({len(inlet_p)} inlet pts, "
\t\t\t\tf"{len(outlet_p)} outlet pts) -- grid resolution may be "
\t\t\t\tf"too coarse in r"
\t\t\t),
\t\t}

\tinlet_avg = sum(inlet_p) / len(inlet_p)
\toutlet_avg = sum(outlet_p) / len(outlet_p)
\tkind = (
\t\t"inlet=total(static+dynamic), outlet=static-only"
\t\tif rho_kgm3 is not None else "static-only (both sides)"
\t)
\treturn {
\t\t"inlet_pressure_pa": inlet_avg,
\t\t"outlet_pressure_pa": outlet_avg,
\t\t"pressure_drop_pa": inlet_avg - outlet_avg,
\t\t"detail": (
\t\t\tf"{len(inlet_p)} inlet pts, {len(outlet_p)} outlet pts at z={z0} "
\t\t\tf"({kind} pressure)"
\t\t),
\t}


'''

# Locate function boundary: from "def compute_pressure_drop(" up to the
# next top-level "def " (mass_conservation_metrics).
pattern = re.compile(r"def compute_pressure_drop\(.*?\n\n\ndef mass_conservation_metrics", re.DOTALL)
match = pattern.search(sc)
if not match:
    raise RuntimeError(
        "Could not locate compute_pressure_drop function boundary in "
        f"{sc_path} -- file structure may differ from expected. Aborting "
        "patch to avoid corrupting the file."
    )

sc_patched = sc[:match.start()] + NEW_FUNC + "def mass_conservation_metrics" + sc[match.end():]
with open(sc_path, "w", encoding="utf-8") as f:
    f.write(sc_patched)

print(f"[patched] {sc_path}")

# ── 2. Patch validate_pressure_drop.py: force RATIO_WARN_LOW/HIGH back
#      to 0.5/2.0 regardless of current value, and remove any
#      EXPECTED_RATIO leftovers from the summary text. ─────────────────
vp_path = f"{BASE}/validate_pressure_drop.py"
with open(vp_path, "r", encoding="utf-8") as f:
    vp = f.read()

vp_patched = re.sub(
    r"RATIO_WARN_LOW = .*?\nRATIO_WARN_HIGH = .*?\n",
    "RATIO_WARN_LOW = 0.5\nRATIO_WARN_HIGH = 2.0\n",
    vp,
    count=1,
)

vp_patched = re.sub(
    r'print\(\s*f?"A ratio consistently.*?\)\s*\n',
    (
        'print(\n'
        '            "A ratio consistently near 1.0 across very different designs means "\n'
        '            "the field solve and baseline agree; a consistent OFFSET (all ratios "\n'
        '            "clustered around some other constant, e.g. ~0.45 or ~2.2) points to "\n'
        '            "a remaining systematic/definitional issue rather than per-design "\n'
        '            "under-training; a WIDE, inconsistent spread instead points to "\n'
        '            "per-design training quality (extrapolation, under-trained regions "\n'
        '            "of the (D,Q) window) rather than a single fixed bug."\n'
        '        )\n'
    ),
    vp_patched,
    count=1,
    flags=re.DOTALL,
)

with open(vp_path, "w", encoding="utf-8") as f:
    f.write(vp_patched)

print(f"[patched] {vp_path}")

# ── 3. Verify ─────────────────────────────────────────────────────────
with open(sc_path) as f:
    sc_check = f.read()
with open(vp_path) as f:
    vp_check = f.read()

assert 'outlet_p = [p for (r, vr, vt, vz, p) in pts if r < r_exhaust_m]' in sc_check, \
    "VERIFY FAILED: sanity_check.py outlet line not as expected"
assert "RATIO_WARN_LOW = 0.5" in vp_check, \
    "VERIFY FAILED: validate_pressure_drop.py bounds not reset"
assert "EXPECTED_RATIO" not in vp_check, \
    "VERIFY FAILED: EXPECTED_RATIO leftovers still present"

print("\n✅ VERIFIED: both files now contain the correct, final code.")
print("Now run: !python validate_pressure_drop.py /content/drive/MyDrive/cyclone_model_parametric.pth")