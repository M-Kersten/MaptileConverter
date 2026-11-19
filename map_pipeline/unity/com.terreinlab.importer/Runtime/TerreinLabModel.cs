using UnityEngine;

namespace TerreinLab
{
    /// <summary>
    /// Ties an imported TerreinLab model to the Dutch national grid: RD New (EPSG:28992) across
    /// and NAP heights up.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The TerreinLab importer puts this on the root of every TerreinLab model and fills it from the
    /// <c>metadata.json</c> beside the FBX. The model is built in metres around the middle of its
    /// area, with Bake Axis Conversion on, so in the root's own frame
    /// </para>
    /// <code>
    /// rd_x = local_x + origin_rd[0]
    /// rd_y = local_z + origin_rd[1]
    /// nap  = local_y + ground_z_offset_nap
    /// </code>
    /// <para>
    /// The world methods go through this object's transform first, so they stay right when the
    /// model is moved, turned or scaled in a scene. The local methods do not.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var model = GetComponentInParent&lt;TerreinLabModel&gt;();
    /// RdNapPosition here = model.WorldToRdNap(transform.position);
    /// Vector3 marker = model.RdNapToWorld(136900.0, 455850.0, 5.2);
    /// </code>
    /// </example>
    [DisallowMultipleComponent]
    [AddComponentMenu("TerreinLab/TerreinLab Model")]
    public sealed class TerreinLabModel : MonoBehaviour
    {
        /// <summary>The coordinate reference system every TerreinLab model is built in.</summary>
        public const string RdCrs = "EPSG:28992";

        [SerializeField]
        [Tooltip("The area name the model was built under (\"name\" in metadata.json).")]
        string areaName = string.Empty;

        [SerializeField]
        [Tooltip("RD easting of the model's local origin, in metres (origin_rd[0]).")]
        double originRdX;

        [SerializeField]
        [Tooltip("RD northing of the model's local origin, in metres (origin_rd[1]).")]
        double originRdY;

        [SerializeField]
        [Tooltip("NAP height of the model's local Y = 0, in metres (ground_z_offset_nap).")]
        double groundZOffsetNap;

        [SerializeField]
        [Tooltip("West edge of the area in RD metres (bbox_rd[0]).")]
        double bboxRdXMin = double.NaN;

        [SerializeField]
        [Tooltip("South edge of the area in RD metres (bbox_rd[1]).")]
        double bboxRdYMin = double.NaN;

        [SerializeField]
        [Tooltip("East edge of the area in RD metres (bbox_rd[2]).")]
        double bboxRdXMax = double.NaN;

        [SerializeField]
        [Tooltip("North edge of the area in RD metres (bbox_rd[3]).")]
        double bboxRdYMax = double.NaN;

        /// <summary>The area name the model was built under, as <c>name</c> in <c>metadata.json</c>.</summary>
        public string AreaName => areaName;

        /// <summary>RD easting of the model's local origin in metres: <c>origin_rd[0]</c>.</summary>
        public double OriginRdX => originRdX;

        /// <summary>RD northing of the model's local origin in metres: <c>origin_rd[1]</c>.</summary>
        public double OriginRdY => originRdY;

        /// <summary>
        /// The NAP height in metres of local Y = 0: <c>ground_z_offset_nap</c>. Every height in the
        /// model had this subtracted, so add it back to get NAP.
        /// </summary>
        public double GroundZOffsetNap => groundZOffsetNap;

        /// <summary>West edge of the built area in RD metres, or NaN when it is not known.</summary>
        public double BboxRdXMin => bboxRdXMin;

        /// <summary>South edge of the built area in RD metres, or NaN when it is not known.</summary>
        public double BboxRdYMin => bboxRdYMin;

        /// <summary>East edge of the built area in RD metres, or NaN when it is not known.</summary>
        public double BboxRdXMax => bboxRdXMax;

        /// <summary>North edge of the built area in RD metres, or NaN when it is not known.</summary>
        public double BboxRdYMax => bboxRdYMax;

        /// <summary>
        /// Sets what the model was built with. The importer calls this; call it yourself only for a
        /// model that did not come through the importer.
        /// </summary>
        /// <param name="name">The area name.</param>
        /// <param name="originX">RD easting of the local origin, <c>origin_rd[0]</c>.</param>
        /// <param name="originY">RD northing of the local origin, <c>origin_rd[1]</c>.</param>
        /// <param name="groundOffsetNap">NAP height of local Y = 0, <c>ground_z_offset_nap</c>.</param>
        /// <param name="xMin">West edge of the area in RD, or NaN.</param>
        /// <param name="yMin">South edge of the area in RD, or NaN.</param>
        /// <param name="xMax">East edge of the area in RD, or NaN.</param>
        /// <param name="yMax">North edge of the area in RD, or NaN.</param>
        public void Configure(
            string name,
            double originX,
            double originY,
            double groundOffsetNap,
            double xMin = double.NaN,
            double yMin = double.NaN,
            double xMax = double.NaN,
            double yMax = double.NaN)
        {
            areaName = name ?? string.Empty;
            originRdX = originX;
            originRdY = originY;
            groundZOffsetNap = groundOffsetNap;
            bboxRdXMin = xMin;
            bboxRdYMin = yMin;
            bboxRdXMax = xMax;
            bboxRdYMax = yMax;
        }

        /// <summary>
        /// RD and NAP of a position given in the model root's own frame: X east, Y up, Z north, in
        /// metres from the middle of the area.
        /// </summary>
        /// <param name="localPosition">A position relative to this object, not to the world.</param>
        public RdNapPosition LocalToRdNap(Vector3 localPosition)
        {
            return new RdNapPosition(
                localPosition.x + originRdX,
                localPosition.z + originRdY,
                localPosition.y + groundZOffsetNap);
        }

        /// <summary>
        /// The position in the model root's own frame of an RD and NAP coordinate. The inverse of
        /// <see cref="LocalToRdNap"/>.
        /// </summary>
        /// <param name="rdX">RD easting in metres.</param>
        /// <param name="rdY">RD northing in metres.</param>
        /// <param name="nap">Height above NAP in metres.</param>
        public Vector3 RdNapToLocal(double rdX, double rdY, double nap)
        {
            // Subtract in double first: the differences are small enough for a float, the RD
            // values themselves are not.
            return new Vector3(
                (float)(rdX - originRdX),
                (float)(nap - groundZOffsetNap),
                (float)(rdY - originRdY));
        }

        /// <summary>
        /// RD and NAP of a Unity world position, through this object's transform, so it holds
        /// wherever the model has been moved, turned or scaled to.
        /// </summary>
        /// <param name="worldPosition">A position in world space, such as <c>transform.position</c>.</param>
        public RdNapPosition WorldToRdNap(Vector3 worldPosition)
        {
            return LocalToRdNap(transform.InverseTransformPoint(worldPosition));
        }

        /// <summary>
        /// The Unity world position of an RD and NAP coordinate, through this object's transform.
        /// The inverse of <see cref="WorldToRdNap"/>.
        /// </summary>
        /// <param name="rdX">RD easting in metres.</param>
        /// <param name="rdY">RD northing in metres.</param>
        /// <param name="nap">Height above NAP in metres.</param>
        public Vector3 RdNapToWorld(double rdX, double rdY, double nap)
        {
            return transform.TransformPoint(RdNapToLocal(rdX, rdY, nap));
        }

        /// <summary>The Unity world position of <paramref name="position"/>.</summary>
        /// <param name="position">An RD and NAP position.</param>
        public Vector3 RdNapToWorld(RdNapPosition position)
        {
            return RdNapToWorld(position.RdX, position.RdY, position.Nap);
        }

        /// <summary>The NAP height in metres of a Unity world position.</summary>
        /// <param name="worldPosition">A position in world space.</param>
        public double WorldToNap(Vector3 worldPosition)
        {
            return WorldToRdNap(worldPosition).Nap;
        }

        /// <summary>
        /// Whether an RD coordinate lies inside the area the model was built for. False when the
        /// area is not known.
        /// </summary>
        /// <param name="rdX">RD easting in metres.</param>
        /// <param name="rdY">RD northing in metres.</param>
        public bool ContainsRd(double rdX, double rdY)
        {
            // Comparisons with NaN are false, so an unknown area contains nothing.
            return rdX >= bboxRdXMin && rdX <= bboxRdXMax && rdY >= bboxRdYMin && rdY <= bboxRdYMax;
        }
    }
}
