using System;
using System.Collections.Generic;

namespace TerreinLab.Importing
{
    /// <summary>
    /// The fields of a TerreinLab <c>metadata.json</c> the importer uses: the documented contract
    /// that walks a local position back to RD and NAP.
    /// </summary>
    /// <remarks>
    /// The file holds much more (terrain statistics, sources, materials, attribution), all of it
    /// ignored here. What marks a file as TerreinLab's is the combination of <c>crs</c> set to
    /// <c>EPSG:28992</c>, <c>origin_rd</c> as two numbers and <c>ground_z_offset_nap</c> as a
    /// number: every version of the pipeline has written all three, and the importer cannot place
    /// the model without them.
    /// </remarks>
    internal sealed class TerreinLabMetadata
    {
        TerreinLabMetadata()
        {
        }

        /// <summary>The area name, or an empty string when the file has none.</summary>
        public string Name { get; private set; } = string.Empty;

        /// <summary>The coordinate reference system; always <see cref="TerreinLabNames.RdCrs"/>.</summary>
        public string Crs { get; private set; } = string.Empty;

        /// <summary>RD easting of the model's local origin, <c>origin_rd[0]</c>.</summary>
        public double OriginRdX { get; private set; }

        /// <summary>RD northing of the model's local origin, <c>origin_rd[1]</c>.</summary>
        public double OriginRdY { get; private set; }

        /// <summary>NAP height of local Y = 0, <c>ground_z_offset_nap</c>.</summary>
        public double GroundZOffsetNap { get; private set; }

        /// <summary>West edge of the area in RD, or NaN when <c>bbox_rd</c> is missing.</summary>
        public double BboxRdXMin { get; private set; } = double.NaN;

        /// <summary>South edge of the area in RD, or NaN when <c>bbox_rd</c> is missing.</summary>
        public double BboxRdYMin { get; private set; } = double.NaN;

        /// <summary>East edge of the area in RD, or NaN when <c>bbox_rd</c> is missing.</summary>
        public double BboxRdXMax { get; private set; } = double.NaN;

        /// <summary>North edge of the area in RD, or NaN when <c>bbox_rd</c> is missing.</summary>
        public double BboxRdYMax { get; private set; } = double.NaN;

        /// <summary>The axis convention the file states, or null when it states none.</summary>
        public string UnityAxis { get; private set; }

        /// <summary>
        /// Whether the axis convention is the one <see cref="TerreinLab.TerreinLabModel"/> converts
        /// with. True when the file does not say.
        /// </summary>
        public bool AxisAsExpected => UnityAxis == null || UnityAxis == TerreinLabNames.UnityAxis;

        /// <summary>Reads the fields from the text of a <c>metadata.json</c>.</summary>
        /// <param name="json">The whole file.</param>
        /// <param name="metadata">The fields, or null when the text is not TerreinLab metadata.</param>
        /// <param name="reason">Why it is not, or null.</param>
        /// <returns>True when the text is TerreinLab metadata.</returns>
        public static bool TryParse(string json, out TerreinLabMetadata metadata, out string reason)
        {
            metadata = null;
            if (!MiniJson.TryParse(json, out object root, out string error))
            {
                reason = "not valid JSON: " + error;
                return false;
            }
            if (!(root is Dictionary<string, object> fields))
            {
                reason = "not a JSON object";
                return false;
            }

            if (!(Get(fields, TerreinLabNames.JsonCrs) is string crs)
                || !string.Equals(crs, TerreinLabNames.RdCrs, StringComparison.OrdinalIgnoreCase))
            {
                reason = "\"" + TerreinLabNames.JsonCrs + "\" is not " + TerreinLabNames.RdCrs;
                return false;
            }
            if (!TryGetNumbers(fields, TerreinLabNames.JsonOriginRd, 2, out double[] origin))
            {
                reason = "\"" + TerreinLabNames.JsonOriginRd + "\" is not two numbers";
                return false;
            }
            if (!(Get(fields, TerreinLabNames.JsonGroundZOffsetNap) is double groundOffset)
                || double.IsNaN(groundOffset) || double.IsInfinity(groundOffset))
            {
                reason = "\"" + TerreinLabNames.JsonGroundZOffsetNap + "\" is not a number";
                return false;
            }

            metadata = new TerreinLabMetadata
            {
                Name = Get(fields, TerreinLabNames.JsonName) as string ?? string.Empty,
                Crs = crs,
                OriginRdX = origin[0],
                OriginRdY = origin[1],
                GroundZOffsetNap = groundOffset,
                UnityAxis = Get(fields, TerreinLabNames.JsonUnityAxis) as string,
            };
            if (TryGetNumbers(fields, TerreinLabNames.JsonBboxRd, 4, out double[] bbox))
            {
                metadata.BboxRdXMin = bbox[0];
                metadata.BboxRdYMin = bbox[1];
                metadata.BboxRdXMax = bbox[2];
                metadata.BboxRdYMax = bbox[3];
            }
            reason = null;
            return true;
        }

        static object Get(Dictionary<string, object> fields, string key)
        {
            return fields.TryGetValue(key, out object value) ? value : null;
        }

        static bool TryGetNumbers(Dictionary<string, object> fields, string key, int count, out double[] numbers)
        {
            numbers = null;
            if (!(Get(fields, key) is List<object> list) || list.Count != count)
                return false;
            var result = new double[count];
            for (int i = 0; i < count; i++)
            {
                if (!(list[i] is double number) || double.IsNaN(number) || double.IsInfinity(number))
                    return false;
                result[i] = number;
            }
            numbers = result;
            return true;
        }
    }
}
