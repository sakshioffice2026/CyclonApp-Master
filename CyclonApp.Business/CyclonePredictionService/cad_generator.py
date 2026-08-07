import subprocess
import os


def generate_cyclone_cad(params: dict, output_dir: str) -> dict:
    try:
        revision_id = params['RevisionId']
        barrel_dia = params['BarrelDiameterMm']
        barrel_height = params['BarrelHeightMm']
        cone_height = params['ConeHeightMm']
        exhaust_dia = params['ExhaustDiaMm']
        exhaust_length = params['ExhaustLengthMm']
        outlet_dia = params['BottomOutletMm']
        inlet_height = params['InletHeightMm']
        inlet_width = params['InletWidthMm']

        os.makedirs(output_dir, exist_ok=True)

        step_file = f"cyclone_rev{revision_id}.step"
        dxf_file = f"cyclone_rev{revision_id}.dxf"
        step_path = os.path.join(output_dir, step_file)
        dxf_path = os.path.join(output_dir, dxf_file)

        macro_content = f"""
import Part
import importDXF
import FreeCAD as App

doc = App.newDocument("Cyclone")

barrel = Part.makeCylinder({barrel_dia}/2, {barrel_height})
cone = Part.makeCone({barrel_dia}/2, {outlet_dia}/2, {cone_height})
cone.translate(App.Vector(0, 0, -{cone_height}))
body = barrel.fuse(cone)

exhaust_pipe = Part.makeCylinder({exhaust_dia}/2, {exhaust_length}, App.Vector(0, 0, {barrel_height}))
exhaust_bore = Part.makeCylinder({exhaust_dia}/2, {exhaust_length} + {barrel_height}, App.Vector(0, 0, 0))
body = body.cut(exhaust_bore)
body = body.fuse(exhaust_pipe)

inlet = Part.makeBox(
    {inlet_width}, {inlet_width}, {inlet_height},
    App.Vector({barrel_dia}/2 - {inlet_width}/2, -{barrel_dia}/2, {barrel_height} - {inlet_height})
)
body = body.fuse(inlet)

obj = doc.addObject("Part::Feature", "Cyclone")
obj.Shape = body
doc.recompute()

Part.export([obj], "{step_path}")

import Drawing
importDXF.export([obj], "{dxf_path}")

doc.close()
print("SUCCESS")
"""

        macro_path = os.path.join(output_dir, "macro.FCMacro")
        with open(macro_path, 'w') as f:
            f.write(macro_content)

        freecad_path = r"C:\Program Files\FreeCAD 1.1\bin\freecad.exe"

        result = subprocess.run(
            [freecad_path, "--console", "-M", macro_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        os.unlink(macro_path)

        if result.returncode != 0:
            raise Exception(result.stderr)

        return {
            "StepUrl": f"/cad-exports/{revision_id}/{step_file}",
            "DxfUrl": f"/cad-exports/{revision_id}/{dxf_file}",
            "PdfUrl": None,
            "Success": True
        }
    except Exception as e:
        raise Exception(f"CAD generation failed: {str(e)}")