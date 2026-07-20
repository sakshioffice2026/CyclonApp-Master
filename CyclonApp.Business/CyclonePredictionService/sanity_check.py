"""
sanity_check.py
────────────────
Standalone physical sanity checks for a field_train.py grid result JSON
(the file written by --output-json). Run this after any training run to
check the result isn't just numerically stable (small loss) but actually
physically sane — a small loss number alone doesn't guarantee that.

Usage:
	python sanity_check.py full_run.json [--v-inlet 157.3156]

--v-inlet is optional but recommended: pass the same v_inlet the run
printed (e.g. from the "Done" summary or the JSON's v_inlet_ms field) so
the velocity-magnitude check has something to compare against. If omitted,
the script reads v_inlet_ms from the JSON itself when present.

Checks performed, each printed as PASS/WARN/FAIL:
  1. No NaN/Inf anywhere in the result.
  2. Velocity magnitudes are within a sane multiple of v_inlet (a trained
	 field showing internal speeds 5-10x the inlet speed is a red flag,
	 not a "strong swirl").
  3. Near-wall velocities (outermost ~2% of r at each z) are close to
	 zero — the no-slip wall boundary condition should hold, not just be
	 "in the loss" with some residual weight.
  4. Near-axis radial/tangential velocities (innermost ~2% of r) are
	 close to zero — the axis symmetry condition should hold.
  5. Axial volumetric flow Q(z) = integral of v_z * 2*pi*r dr, computed at
	 several z cross-sections and checked for consistency with each other.
	 This is a stronger, independent check than the training loss: it
	 verifies mass conservation is actually satisfied in aggregate, not
	 just that the pointwise continuity residual was small at randomly
	 sampled collocation points.

This script only reads the grid the training run already wrote — it does
not reload the model or require torch installed.
"""
import argparse
import json
import sys


def _color(ok: str) -> str:
	return {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[ok]


def report(name: str, status: str, detail: str) -> None:
	print(f"[{_color(status):4s}] {name}: {detail}")


def load_grid(path: str) -> dict:
	with open(path, "r") as f:
		data = json.load(f)
	# Support both the CLI's flat payload ({"grid": {...}, "v_inlet_ms": ...})
	# and a raw grid dict passed directly.
	if "grid" in data:
		grid = data["grid"]
		grid["_v_inlet_ms"] = data.get("v_inlet_ms")
		grid["_q_design_m3s"] = data.get("q_design_m3s")
	else:
		grid = data
		grid["_v_inlet_ms"] = None
		grid["_q_design_m3s"] = None
	return grid


def check_nan_inf(grid: dict) -> bool:
	import math
	bad_fields = []
	for key in ("v_r_ms", "v_theta_ms", "v_z_ms", "pressure_pa"):
		values = grid.get(key, [])
		if any((v != v) or math.isinf(v) for v in values):  # v != v catches NaN
			bad_fields.append(key)
	if bad_fields:
		report("NaN/Inf check", "FAIL", f"found NaN or Inf in: {', '.join(bad_fields)}")
		return False
	report("NaN/Inf check", "PASS", "no NaN or Inf values in v_r, v_theta, v_z, or pressure")
	return True


def check_velocity_magnitude(grid: dict, v_inlet: float) -> bool:
	v_r, v_theta, v_z = grid["v_r_ms"], grid["v_theta_ms"], grid["v_z_ms"]
	max_speed = max(
		(vr ** 2 + vt ** 2 + vz ** 2) ** 0.5
		for vr, vt, vz in zip(v_r, v_theta, v_z)
	)
	ratio = max_speed / v_inlet if v_inlet else float("nan")
	if ratio > 5.0:
		report("Velocity magnitude", "FAIL",
			   f"max internal speed {max_speed:.2f} m/s is {ratio:.1f}x v_inlet "
			   f"({v_inlet:.2f} m/s) — implausibly high, likely still unstable")
		return False
	if ratio > 2.5:
		report("Velocity magnitude", "WARN",
			   f"max internal speed {max_speed:.2f} m/s is {ratio:.1f}x v_inlet "
			   f"({v_inlet:.2f} m/s) — plausible for strong swirl acceleration "
			   f"near the exhaust, but worth a second look")
		return True
	report("Velocity magnitude", "PASS",
		   f"max internal speed {max_speed:.2f} m/s is {ratio:.2f}x v_inlet "
		   f"({v_inlet:.2f} m/s) — within a sane range")
	return True


def _group_by_z(grid: dict) -> dict:
	"""Groups (r, v_r, v_theta, v_z, p) by exact z value — safe because z
	values come from a fixed linspace and are repeated exactly across the
	grid, not independently sampled."""
	groups: dict[float, list[tuple]] = {}
	for r, z, vr, vt, vz, p in zip(
		grid["r_m"], grid["z_m"], grid["v_r_ms"], grid["v_theta_ms"],
		grid["v_z_ms"], grid["pressure_pa"],
	):
		groups.setdefault(z, []).append((r, vr, vt, vz, p))
	return groups


def check_wall_and_axis(grid: dict, v_inlet: float) -> bool:
	groups = _group_by_z(grid)

	wall_speeds = []
	axis_vr = []
	axis_vtheta = []

	for z, pts in groups.items():

		# Skip inlet plane (z = 0)
		if abs(z) < 1e-8:
			continue

		pts_sorted = sorted(pts, key=lambda p: p[0])

		r_max_here = pts_sorted[-1][0]
		if r_max_here <= 0:
			continue

		near_wall_cut = 0.98 * r_max_here
		near_axis_cut = 0.02 * r_max_here

		for r, vr, vt, vz, p in pts_sorted:

			if r >= near_wall_cut:
				wall_speeds.append((vr**2 + vt**2 + vz**2)**0.5)

			if r <= near_axis_cut:
				axis_vr.append(abs(vr))
				axis_vtheta.append(abs(vt))

	ok = True

	if wall_speeds:
		max_wall_speed = max(wall_speeds)
		ratio = max_wall_speed / v_inlet

		if ratio > 0.15:
			report(
				"Wall no-slip",
				"FAIL",
				f"max near-wall speed {max_wall_speed:.3f} m/s "
				f"({ratio:.1%} of v_inlet) — wall BC not well satisfied",
			)
			ok = False

		elif ratio > 0.05:
			report(
				"Wall no-slip",
				"WARN",
				f"max near-wall speed {max_wall_speed:.3f} m/s "
				f"({ratio:.1%} of v_inlet)",
			)

		else:
			report(
				"Wall no-slip",
				"PASS",
				f"max near-wall speed {max_wall_speed:.3f} m/s "
				f"({ratio:.1%} of v_inlet)",
			)

	if axis_vr and axis_vtheta:

		max_axis = max(max(axis_vr), max(axis_vtheta))
		ratio = max_axis / v_inlet

		if ratio > 0.15:
			report(
				"Axis symmetry",
				"FAIL",
				f"max near-axis |v_r|/|v_theta| {max_axis:.3f} m/s "
				f"({ratio:.1%} of v_inlet)",
			)
			ok = False

		elif ratio > 0.05:
			report(
				"Axis symmetry",
				"WARN",
				f"max near-axis |v_r|/|v_theta| {max_axis:.3f} m/s "
				f"({ratio:.1%} of v_inlet)",
			)

		else:
			report(
				"Axis symmetry",
				"PASS",
				f"max near-axis |v_r|/|v_theta| {max_axis:.3f} m/s "
				f"({ratio:.1%} of v_inlet)",
			)

	return ok


def mass_conservation_metrics(grid: dict, q_design: float | None = None) -> dict:
	"""Silent, structured version of the Q(z) mass-conservation check —
	same math as check_mass_conservation below, but returns a dict instead
	of printing, so callers that need the number (not just a console line)
	can use it. field_train.run_field_prediction_job calls this to attach
	massConservationStatus/massFlowSpread to the job result; sanity_check's
	own check_mass_conservation (CLI/PASS-FAIL reporting) now delegates
	here too, so the two can never drift apart.

	Returns:
		{"status": "ok" | "warning" | "failed" | "unknown",
		 "rel_spread": float | None,
		 "detail": str}
		"unknown" means too few z-levels/cross-sections to judge either way
		(not a pass or a fail — just not enough data).
	"""
	import math
	groups = _group_by_z(grid)
	z_values = sorted(groups.keys())
	if len(z_values) < 3:
		return {"status": "unknown", "rel_spread": None, "detail": "too few z-levels to check"}

	flows = []
	for z in z_values:
		pts_sorted = sorted(groups[z], key=lambda p: p[0])
		rs = [p[0] for p in pts_sorted]
		vzs = [p[3] for p in pts_sorted]
		if len(rs) < 2:
			continue
		q = 0.0
		for i in range(1, len(rs)):
			r0, r1 = rs[i - 1], rs[i]
			integrand0 = vzs[i - 1] * 2 * math.pi * r0
			integrand1 = vzs[i] * 2 * math.pi * r1
			q += 0.5 * (integrand0 + integrand1) * (r1 - r0)
		flows.append((z, q))

	if len(flows) < 3:
		return {"status": "unknown", "rel_spread": None, "detail": "too few valid cross-sections to check"}

	# Skip the first/last few z-levels — inlet/outlet regions have real
	# inflow/outflow so Q(z) is EXPECTED to change there; the barrel's
	# mid-section (away from inlet and outlets) is where Q(z) should be
	# closest to constant if continuity is well satisfied.
	mid = flows[len(flows) // 4 : 3 * len(flows) // 4]
	if len(mid) < 2:
		mid = flows

	q_values = [q for _, q in mid]
	q_mean = sum(q_values) / len(q_values)
	q_spread = max(q_values) - min(q_values)

	# Characteristic scale for the relative-spread check:
	# - When |mean Q| is appreciable, use |mean| (original check) so a
	#   mid-section that drains from ~0.03→0.01 still FAILs hard.
	# - When |mean Q|≈0 (correct reverse-flow answer), fall back to a
	#   fraction of design flow so we do not divide by ~0 and false-fail.
	# Do NOT fold in peak |Q| from the inlet plane — that dwarfs the
	# mid-section signal and hides real drain failures.
	if q_design is None:
		q_design = grid.get("_q_design_m3s")
	design_abs = abs(float(q_design)) if q_design is not None else 0.0
	q_floor = max(0.05 * design_abs, 1e-6)
	q_char = max(abs(q_mean), q_floor)
	rel_spread = q_spread / q_char
	level_frac = abs(q_mean) / max(design_abs, q_floor)

	if rel_spread > 0.5:
		return {
			"status": "failed",
			"rel_spread": rel_spread,
			"detail": (
				f"volumetric flow Q(z) varies by {rel_spread:.1%} of "
				f"characteristic scale {q_char:.4f} m^3/s across the barrel "
				f"mid-section (mean {q_mean:.4f} m^3/s) — continuity not well "
				f"satisfied in aggregate"
			),
		}
	elif rel_spread > 0.2:
		return {
			"status": "warning",
			"rel_spread": rel_spread,
			"detail": (
				f"Q(z) varies by {rel_spread:.1%} of characteristic scale "
				f"{q_char:.4f} m^3/s across the barrel mid-section "
				f"(mean {q_mean:.4f} m^3/s) — some inconsistency, may improve "
				f"with more training"
			),
		}
	else:
		extra = ""
		if level_frac > 0.5:
			extra = (
				f" Note: |mean Q| is {level_frac:.0%} of scale — large "
				f"net through-flow; reverse-flow exhaust may be weak."
			)
		return {
			"status": "ok",
			"rel_spread": rel_spread,
			"detail": (
				f"Q(z) varies by only {rel_spread:.1%} of characteristic scale "
				f"{q_char:.4f} m^3/s across the barrel mid-section "
				f"(mean {q_mean:.4f} m^3/s) — continuity well satisfied in "
				f"aggregate.{extra}"
			),
		}


def check_mass_conservation(grid: dict, q_design: float | None = None) -> bool:
	"""Computes Q(z) = integral of v_z * 2*pi*r dr at each z cross-section
	(trapezoidal rule over the sorted r values present at that z) and
	checks Q(z) stays reasonably consistent across sections. This is
	stronger evidence of a physically valid solution than the training
	loss alone — it checks continuity is satisfied in aggregate, not just
	that the pointwise PDE residual was small at the collocation points
	used during training.

	q_design: optional design volumetric flow (m^3/s). When provided (or
	recoverable from the JSON), relative spread is measured against this
	stable scale rather than mean(Q). That matters for reverse-flow
	cyclones, where the physically correct mid-plane net Q is ~0 — dividing
	by mean(Q) then false-fails even a good solution.

	Thin PASS/WARN/FAIL-printing wrapper around mass_conservation_metrics —
	see that function for the actual computation.
	"""
	groups = _group_by_z(grid)
	z_values = sorted(groups.keys())
	metrics = mass_conservation_metrics(grid, q_design=q_design)

	if metrics["status"] == "unknown":
		report("Mass conservation", "WARN", metrics["detail"])
		return True

	if len(z_values) >= 3:
		print("\nQ(z) mid-section relative spread computed — see detail below.")

	status_map = {"failed": "FAIL", "warning": "WARN", "ok": "PASS"}
	report("Mass conservation", status_map[metrics["status"]], metrics["detail"])
	return metrics["status"] != "failed"
def main():
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("json_path", help="Path to the field_train.py --output-json result")
	p.add_argument("--v-inlet", type=float, default=None,
					help="Inlet velocity (m/s) to compare against. If omitted, "
						 "read from the JSON's v_inlet_ms field when present.")
	args = p.parse_args()

	grid = load_grid(args.json_path)
	v_inlet = args.v_inlet if args.v_inlet is not None else grid.get("_v_inlet_ms")
	if v_inlet is None:
		print("ERROR: no --v-inlet given and none found in the JSON "
			  "(v_inlet_ms field). Pass --v-inlet explicitly.", file=sys.stderr)
		sys.exit(1)

	print(f"Checking {args.json_path}  (v_inlet = {v_inlet:.4f} m/s, "
		  f"{len(grid['r_m'])} grid points)\n")

	results = [
		check_nan_inf(grid),
		check_velocity_magnitude(grid, v_inlet),
		check_wall_and_axis(grid, v_inlet),
		check_mass_conservation(grid, q_design=grid.get("_q_design_m3s")),
	]

	print()
	if all(results):
		print("Overall: all checks passed or warned only — result looks physically sane.")
	else:
		print("Overall: at least one check FAILED — do not trust this result yet.")


if __name__ == "__main__":
	main()