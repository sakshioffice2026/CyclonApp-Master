using System;
using System.Collections.Generic;

namespace CyclonApp.Business.CyclonePrediction
{
    /// <summary>
    /// Builds the same regular (r, z) evaluation grid, filtered to the
    /// fluid domain, that field_model.py's evaluate_grid() builds on the
    /// Python side — kept in sync with field_physics.CycloneAxisymGeometry
    /// so the ONNX (CPU, synchronous) path and the Python (async job) path
    /// query the network over an equivalent domain.
    ///
    /// Mirrors CycloneAxisymGeometry exactly:
    ///   - outer_wall_radius: constant r_barrel through the barrel, then a
    ///     linear taper down to r_bottom_outlet through the cone.
    ///   - is_fluid: r in [0, wall_r] and z in [0, total_height].
    ///
    /// All inputs/outputs in meters.
    /// </summary>
    public static class CycloneFieldGridBuilder
    {
        public sealed class Grid
        {
            public float[] R { get; init; } = Array.Empty<float>();
            public float[] Z { get; init; } = Array.Empty<float>();
        }

        public static Grid Build(
            double barrelDiameterM,
            double barrelHeightM,
            double coneHeightM,
            double bottomOutletM,
            int nR = 40,
            int nZ = 60)
        {
            double rBarrel = barrelDiameterM / 2.0;
            double zBarrelEnd = barrelHeightM;
            double zConeEnd = barrelHeightM + coneHeightM;
            double rBottomOutlet = bottomOutletM / 2.0;
            double totalHeight = zConeEnd;

            var rOut = new List<float>(nR * nZ);
            var zOut = new List<float>(nR * nZ);

            for (int i = 0; i < nR; i++)
            {
                double r = nR == 1 ? 0.0 : rBarrel * i / (nR - 1);
                for (int j = 0; j < nZ; j++)
                {
                    double z = nZ == 1 ? 0.0 : totalHeight * j / (nZ - 1);

                    double wallR;
                    if (z <= zBarrelEnd)
                    {
                        wallR = rBarrel;
                    }
                    else
                    {
                        double coneFrac = (z - zBarrelEnd) / (zConeEnd - zBarrelEnd + 1e-9);
                        coneFrac = Math.Clamp(coneFrac, 0.0, 1.0);
                        wallR = rBarrel + (rBottomOutlet - rBarrel) * coneFrac;
                    }

                    bool isFluid = r >= 0 && r <= wallR && z >= 0 && z <= totalHeight;
                    if (isFluid)
                    {
                        rOut.Add((float)r);
                        zOut.Add((float)z);
                    }
                }
            }

            return new Grid { R = rOut.ToArray(), Z = zOut.ToArray() };
        }
    }
}
