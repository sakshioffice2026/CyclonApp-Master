import subprocess
import os

def generate_cyclone_cad(params: dict, output_dir: str) -> dict:
    try:
        revision_id = params['RevisionId']
        barrel_dia = params['BarrelDiameterMm']
        barrel_height = params['BarrelHeightMm']
        cone_height = params['ConeHeightMm']
        outlet_dia = params['BottomOutletMm']
        
        step_file = f"cyclone_rev{revision_id}.step"
        dxf_file = f"cyclone_rev{revision_id}.dxf"
        step_path = os.path.join(output_dir, step_file)
        dxf_path = os.path.join(output_dir, dxf_file)
        
        macro_content = f"""
import Part
doc = App.newDocument("Cyclone")
barrel = Part.makeCylinder({barrel_dia}/2, {barrel_height})
doc.addObject("Part::Feature", "Barrel").Shape = barrel
cone = Part.makeCone({barrel_dia}/2, {outlet_dia}/2, {cone_height})
cone_obj = doc.addObject("Part::Feature", "Cone")
cone_obj.Shape = cone
cone_obj.Placement.z = -{cone_height}
Part.export(doc.Objects, "{step_path}")
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
            timeout=60
        )
        
        os.unlink(macro_path)
        
        if result.returncode != 0:
            raise Exception(result.stderr)
        
        return {
            "StepUrl": f"/cad/{step_file}",
            "DxfUrl": f"/cad/{dxf_file}",
            "PdfUrl": None,
            "Success": True
        }
    except Exception as e:
        raise Exception(f"CAD generation failed: {str(e)}")