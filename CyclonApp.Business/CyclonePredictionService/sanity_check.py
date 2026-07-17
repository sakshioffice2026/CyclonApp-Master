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
	 several z cross-sections within the constant-diameter barrel region
	 (detected by matching outer radius, not a blind slice of every
	 z-level — the cone's tapering radius makes a naive 25-75% slice of
	 all z-levels unreliable) and checked for consistency with each
	 other. This is a stronger, independent check than the training
	 loss: it verifies mass conservation is actually satisfied in
	 aggregate, not just that the pointwise continuity residual was
	 small at randomly sampled collocation points.

This script only reads the grid the training run already wrote — it does
not reload the model or require torch installed.

FIX (this revision): check_mass_conservation was previously duplicated —
one copy accidentally nested one indent level inside check_wall_and_axis
(so it was never a callable top-level function), and the two copies were
concatenated without a newline between them ("return Truedef
check_mass_conservation..."), which is what produced the IndentationError
at parse time. Replaced with a single, clean, top-level definition below.
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
	else:
		grid = data
		grid["_v_inlet_ms"] = None
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


def check_mass_conservation(grid: dict) -> bool:
	"""Check volumetric flow consistency across the cylindrical barrel.
	Detects the barrel region by matching outer radius (constant-diameter
	section) rather than blindly slicing 25-75% of all z-levels, since the
	cone's tapering radius makes that slice unreliable."""
	import math

	groups = _group_by_z(grid)
	z_values = sorted(groups.keys())

	if len(z_values) < 3:
		report("Mass conservation", "WARN", "too few z-levels to check")
		return True

	flows = []
	max_r_by_z = {}

	for z in z_values:
		pts = sorted(groups[z], key=lambda p: p[0])
		if len(pts) < 2:
			continue
		rs = [p[0] for p in pts]
		vzs = [p[3] for p in pts]
		max_r_by_z[z] = rs[-1]

		q = 0.0
		for i in range(1, len(rs)):
			r0, r1 = rs[i - 1], rs[i]
			vz0, vz1 = vzs[i - 1], vzs[i]
			q += 0.5 * (vz0 * 2.0 * math.pi * r0 + vz1 * 2.0 * math.pi * r1) * (r1 - r0)
		flows.append((z, q))

	if len(flows) < 3:
		report("Mass conservation", "WARN", "too few valid cross-sections to check")
		return True

	# Detect barrel by constant outer radius.
	r_barrel = max(max_r_by_z.values())
	tol = max(1e-6, r_barrel * 0.005)
	barrel_zs = [z for z in z_values if abs(max_r_by_z[z] - r_barrel) <= tol]

	if len(barrel_zs) < 3:
		report("Mass conservation", "WARN", "unable to identify barrel region")
		return True

	start = len(barrel_zs) // 4
	end = 3 * len(barrel_zs) // 4
	mid_zs = set(barrel_zs[start:end] or barrel_zs)
	mid = [(z, q) for z, q in flows if z in mid_zs]
	if len(mid) < 2:
		mid = [(z, q) for z, q in flows if z in barrel_zs]

	print("\nQ(z):")
	for z, q in mid:
		print(f"{z:.4f}  {q:.6f}")

	q_values = [q for _, q in mid]
	q_mean = sum(q_values) / len(q_values)
	rel_spread = (
		(max(q_values) - min(q_values)) / abs(q_mean)
		if abs(q_mean) > 1e-12 else float("inf")
	)

	if rel_spread > 0.5:
		report(
			"Mass conservation", "FAIL",
			f"volumetric flow Q(z) varies by {rel_spread:.1%} "
			f"across barrel mid-section (mean {q_mean:.4f} m^3/s)",
		)
		return False

	if rel_spread > 0.2:
		report(
			"Mass conservation", "WARN",
			f"Q(z) varies by {rel_spread:.1%} "
			f"across barrel mid-section (mean {q_mean:.4f} m^3/s)",
		)
		return True

	report(
		"Mass conservation", "PASS",
		f"Q(z) varies by only {rel_spread:.1%} "
		f"across barrel mid-section (mean {q_mean:.4f} m^3/s)",
	)
	return True


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
		check_mass_conservation(grid),
	]

	print()
	if all(results):
		print("Overall: all checks passed or warned only — result looks physically sane.")
	else:
		print("Overall: at least one check FAILED — do not trust this result yet.")


if __name__ == "__main__":
	main()