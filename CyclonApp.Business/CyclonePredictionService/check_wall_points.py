import json
import math

d = json.load(open("full_run.json"))

g = d["grid"]

r = g["r_m"]
z = g["z_m"]
vr = g["v_r_ms"]
vt = g["v_theta_ms"]
vz = g["v_z_ms"]

rmax = max(r)

pts = []

for i in range(len(r)):
    if r[i] >= 0.98 * rmax:
        speed = math.sqrt(vr[i]**2 + vt[i]**2 + vz[i]**2)
        pts.append((speed, z[i], r[i]))

pts.sort(reverse=True)

for p in pts[:20]:
    print(p)