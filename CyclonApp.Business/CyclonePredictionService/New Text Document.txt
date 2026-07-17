from field_physics import geometry_from_dimensions_mm, inlet_axial_velocity_ms
import torch

geometry = geometry_from_dimensions_mm(
    barrel_diameter_mm=200.0, barrel_height_mm=300.0, cone_height_mm=250.0,
    exhaust_dia_mm=80.0, exhaust_length_mm=90.0, bottom_outlet_mm=40.0,
)
flow_cfm = 500.0
v_z_inlet = inlet_axial_velocity_ms(torch.tensor([flow_cfm]), geometry.r_barrel, geometry.r_exhaust).item()
ring_area = 3.14159265 * (geometry.r_barrel**2 - geometry.r_exhaust**2)
q_check_m3s = v_z_inlet * ring_area
q_target_m3s = flow_cfm * 0.000471947

print(f'v_z_inlet = {v_z_inlet:.4f} m/s')
print(f'reproduced flow = {q_check_m3s:.6f} m^3/s')
print(f'target flow     = {q_target_m3s:.6f} m^3/s')
print(f'match: {abs(q_check_m3s - q_target_m3s) < 1e-9}')