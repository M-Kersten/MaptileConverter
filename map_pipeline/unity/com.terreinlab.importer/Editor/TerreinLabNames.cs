namespace TerreinLab.Importing
{
    /// <summary>
    /// The file names, material name and metadata keys the TerreinLab pipeline writes and this
    /// package keys on.
    /// </summary>
    /// <remarks>
    /// <c>map_pipeline/tests/test_pipeline.py</c> (<c>TestUnityPackage</c>) reads these constants
    /// from this file and checks each against what the pipeline produces, so a rename on either
    /// side fails a test instead of silently leaving models unprocessed. Keep every constant a
    /// plain <c>public const string Name = "value";</c> line so the test can read it.
    /// </remarks>
    internal static class TerreinLabNames
    {
        /// <summary>The model file extension (<c>export.fbx_name</c> is <c>model.fbx</c>).</summary>
        public const string ModelExtension = ".fbx";

        /// <summary>The metadata beside every model (<c>export.metadata_name</c>, <c>src/export.py</c>).</summary>
        public const string MetadataFile = "metadata.json";

        /// <summary>The source credits beside every model (<c>write_attribution</c> in <c>src/export.py</c>).</summary>
        public const string AttributionFile = "ATTRIBUTION.txt";

        /// <summary>The aerial photo the terrain and roofs carry (<c>export.aerial_name</c>).</summary>
        public const string AerialFile = "aerial.png";

        /// <summary>
        /// The start of every roof page, <c>roofs_00.png</c> and up (<c>FILE_STEM</c> in
        /// <c>src/roofs.py</c>): the roofs at a finer resolution than the aerial, packed.
        /// </summary>
        public const string RoofPagePrefix = "roofs_";

        /// <summary>The surface class map, one class per pixel (<c>write_land_cover</c> in <c>src/surfaces.py</c>).</summary>
        public const string LandCoverFile = "landcover.png";

        /// <summary>The suffix of every facade normal map (<c>generate_facade_textures</c> in <c>src/facade.py</c>).</summary>
        public const string NormalMapSuffix = "_normal.png";

        /// <summary>The one material every tree shares (<c>blender/process.py</c>).</summary>
        public const string TreeMaterial = "M_tree";

        /// <summary>The value of <c>crs</c> in the metadata of every TerreinLab model.</summary>
        public const string RdCrs = "EPSG:28992";

        /// <summary>
        /// The value of <c>unity_axis</c> in the metadata: the axis convention the coordinate
        /// conversion in <see cref="TerreinLab.TerreinLabModel"/> assumes.
        /// </summary>
        public const string UnityAxis = "X=east, Y=up(NAP), Z=north";

        /// <summary>Metadata key: the area name.</summary>
        public const string JsonName = "name";

        /// <summary>Metadata key: the coordinate reference system, <see cref="RdCrs"/>.</summary>
        public const string JsonCrs = "crs";

        /// <summary>Metadata key: the RD position of the model's local origin, two numbers.</summary>
        public const string JsonOriginRd = "origin_rd";

        /// <summary>Metadata key: the NAP height of local Y = 0, one number.</summary>
        public const string JsonGroundZOffsetNap = "ground_z_offset_nap";

        /// <summary>Metadata key: the area in RD as xmin, ymin, xmax, ymax.</summary>
        public const string JsonBboxRd = "bbox_rd";

        /// <summary>Metadata key: the axis convention, <see cref="UnityAxis"/>.</summary>
        public const string JsonUnityAxis = "unity_axis";
    }
}
