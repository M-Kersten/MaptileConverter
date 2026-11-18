using System;
using UnityEditor;
using UnityEngine;

namespace TerreinLab.Importing
{
    /// <summary>
    /// The import settings of a TerreinLab model and the textures beside it, in one place, so the
    /// postprocessor and the Reimport menu item apply exactly the same ones.
    /// </summary>
    internal static class TerreinLabImportSettings
    {
        /// <summary>The Max Size the aerial gets: 8192 px, against Unity's default of 2048.</summary>
        public const int AerialMaxSize = 8192;

        /// <summary>The Max Size for an aerial or class map larger than 8192 px, where the platform allows it.</summary>
        public const int LargeMaxSize = 16384;

        /// <summary>
        /// Build targets whose GPUs cannot be relied on for textures over 8192 px. A larger aerial
        /// gets an override of 8192 on these, unless the texture already has an override of its
        /// own there.
        /// </summary>
        public static readonly string[] PlatformsLimitedTo8192 = { "Android", "iPhone", "tvOS", "WebGL" };

        /// <summary>What a texture in a TerreinLab folder is, which decides how it is imported.</summary>
        public enum TextureRole
        {
            /// <summary><c>aerial.png</c>: the photo over the whole area, on the terrain and the roofs.</summary>
            Aerial,

            /// <summary><c>*_normal.png</c>: a facade normal map.</summary>
            NormalMap,

            /// <summary><c>landcover.png</c>: a surface class per pixel, data rather than a picture.</summary>
            LandCover,

            /// <summary>Every other texture: facades, trees, water, furniture, vehicles, rails, structures, fences.</summary>
            Colour,
        }

        /// <summary>
        /// Applies the settings of the README's "Importing into Unity" list to a TerreinLab model.
        /// </summary>
        /// <param name="importer">The model's importer.</param>
        public static void ApplyModelSettings(ModelImporter importer)
        {
            if (importer == null)
                throw new ArgumentNullException(nameof(importer));

            // The model is in metres: Scale Factor 1 with Convert Units on brings every object in at
            // scale 1.
            importer.globalScale = 1f;
            importer.useFileScale = true;

            // Takes the -90 degree X rotation Blender's export leaves on the root objects off and
            // into the geometry, the tree instances included, so every transform reads X east,
            // Y up, Z north. TerreinLabModel's conversion relies on it.
            importer.bakeAxisConversion = true;

            // Lightmap UVs would overwrite the second UV channel, which on the roads is RoadUV.
            importer.generateSecondaryUV = false;

            // The ground has more than 65k vertices and needs 32-bit indices; Auto picks them per
            // mesh and keeps 16-bit where they are enough.
            importer.indexFormat = ModelImporterIndexFormat.Auto;

            // Materials made from the FBX's own descriptions and kept inside the model, each
            // pointing at the textures beside the FBX by name; nothing is copied or extracted.
            // OnPostprocessMaterial only sees materials made this way.
            importer.materialImportMode = ModelImporterMaterialImportMode.ImportViaMaterialDescription;
            importer.materialLocation = ModelImporterMaterialLocation.InPrefab;

            // A static model: no rig, no Animator on the root, no clips. A generic rig over ten
            // thousand objects is a large avatar for nothing.
            importer.animationType = ModelImporterAnimationType.None;
            importer.importAnimation = false;
        }

        /// <summary>What a texture is, from its file name.</summary>
        /// <param name="assetPath">The texture's asset path.</param>
        public static TextureRole RoleOf(string assetPath)
        {
            string file = TerreinLabFolder.FileNameOf(assetPath);
            if (string.Equals(file, TerreinLabNames.AerialFile, StringComparison.OrdinalIgnoreCase))
                return TextureRole.Aerial;
            if (string.Equals(file, TerreinLabNames.LandCoverFile, StringComparison.OrdinalIgnoreCase))
                return TextureRole.LandCover;
            if (file.EndsWith(TerreinLabNames.NormalMapSuffix, StringComparison.OrdinalIgnoreCase))
                return TextureRole.NormalMap;
            return TextureRole.Colour;
        }

        /// <summary>Applies the settings for a texture in a TerreinLab folder, chosen by its name.</summary>
        /// <param name="importer">The texture's importer.</param>
        /// <param name="assetPath">The texture's asset path, read from disk for its size.</param>
        public static void ApplyTextureSettings(TextureImporter importer, string assetPath)
        {
            if (importer == null)
                throw new ArgumentNullException(nameof(importer));

            // All zero when the file is not a PNG, which leaves Max Size where it is.
            PngInfo.TryRead(assetPath, out PngInfo png);

            switch (RoleOf(assetPath))
            {
                case TextureRole.Aerial:
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = true;
                    importer.mipmapEnabled = true;
                    // The photo covers the area exactly. Repeat would bleed the far edge into
                    // this one, and a roof overhanging the area would pick up the other side.
                    importer.wrapMode = TextureWrapMode.Clamp;
                    SetLargeMaxSize(importer, png.LargestSide);
                    break;

                case TextureRole.NormalMap:
                    importer.textureType = TextureImporterType.NormalMap;
                    // The image already is a normal map; converting would read it as heights.
                    importer.convertToNormalmap = false;
                    importer.mipmapEnabled = true;
                    importer.wrapMode = TextureWrapMode.Repeat;
                    RaiseMaxSizeToFit(importer, png.LargestSide);
                    break;

                case TextureRole.LandCover:
                    // Class numbers, not colours: no gamma, no blending between neighbours, no
                    // averaging into mipmaps, no lossy compression, no rescaling to a power of
                    // two, and readable from scripts that ask what surface a point is on.
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = false;
                    importer.alphaSource = TextureImporterAlphaSource.None;
                    importer.mipmapEnabled = false;
                    importer.filterMode = FilterMode.Point;
                    importer.wrapMode = TextureWrapMode.Clamp;
                    importer.npotScale = TextureImporterNPOTScale.None;
                    importer.textureCompression = TextureImporterCompression.Uncompressed;
                    importer.isReadable = true;
                    SetLargeMaxSize(importer, png.LargestSide);
                    break;

                default:
                    importer.textureType = TextureImporterType.Default;
                    importer.sRGBTexture = true;
                    importer.mipmapEnabled = true;
                    // Facades, water, rails and fences tile; the atlases do not mind.
                    importer.wrapMode = TextureWrapMode.Repeat;
                    if (png.HasAlpha)
                    {
                        // The fence: colour is bled into the holes so a filtered edge is not a
                        // dark fringe, and every mipmap keeps as much of it solid at the 0.5
                        // cut-off Blender clips at, so fences do not thin out with distance.
                        importer.alphaSource = TextureImporterAlphaSource.FromInput;
                        importer.alphaIsTransparency = true;
                        importer.mipMapsPreserveCoverage = true;
                        importer.alphaTestReferenceValue = 0.5f;
                    }
                    RaiseMaxSizeToFit(importer, png.LargestSide);
                    break;
            }
        }

        /// <summary>
        /// Max Size 8192, or 16384 when the image is larger than 8192 px, with an 8192 override on
        /// the platforms that may not take more.
        /// </summary>
        static void SetLargeMaxSize(TextureImporter importer, int largestSide)
        {
            bool large = largestSide > AerialMaxSize;
            importer.maxTextureSize = large ? LargeMaxSize : AerialMaxSize;
            if (!large)
                return;
            foreach (string platform in PlatformsLimitedTo8192)
            {
                TextureImporterPlatformSettings settings = importer.GetPlatformTextureSettings(platform);
                // A Max Size chosen for that platform by hand stands.
                if (settings.overridden)
                    continue;
                settings.overridden = true;
                settings.maxTextureSize = AerialMaxSize;
                importer.SetPlatformTextureSettings(settings);
            }
        }

        /// <summary>
        /// Raises Max Size to the image's own size, up to 8192, so a texture is never shrunk below
        /// what the pipeline made; a higher Max Size is left as it is.
        /// </summary>
        static void RaiseMaxSizeToFit(TextureImporter importer, int largestSide)
        {
            int fit = 32;
            while (fit < largestSide && fit < AerialMaxSize)
                fit *= 2;
            if (importer.maxTextureSize < fit)
                importer.maxTextureSize = fit;
        }
    }
}
