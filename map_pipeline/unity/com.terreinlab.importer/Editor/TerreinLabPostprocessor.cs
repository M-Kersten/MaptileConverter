using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace TerreinLab.Importing
{
    /// <summary>
    /// Imports every TerreinLab model, and the textures beside it, with the settings the README
    /// otherwise asks for by hand, and leaves every other asset alone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A TerreinLab model is an <c>.fbx</c> whose folder also holds <c>metadata.json</c> and
    /// <c>ATTRIBUTION.txt</c>, the metadata carrying the RD origin (<see cref="TerreinLabFolder"/>).
    /// For such a model:
    /// </para>
    /// <list type="bullet">
    /// <item><description><see cref="OnPreprocessModel"/> sets the model settings (<see cref="TerreinLabImportSettings.ApplyModelSettings"/>).</description></item>
    /// <item><description><see cref="OnPreprocessTexture"/> sets the aerial, normal map, class map and colour texture settings.</description></item>
    /// <item><description><see cref="OnPostprocessMaterial"/> turns on GPU instancing on every material.</description></item>
    /// <item><description><see cref="OnPostprocessGameObjectWithUserProperties"/> keeps the FBX custom properties on a <see cref="TerreinLabProperties"/> component.</description></item>
    /// <item><description><see cref="OnPostprocessModel"/> puts a <see cref="TerreinLabModel"/> on the root, filled from <c>metadata.json</c>.</description></item>
    /// </list>
    /// <para>
    /// The settings are applied on every import rather than only the first, so a model always comes
    /// out the same whatever its <c>.meta</c> file says.
    /// </para>
    /// <para>
    /// <see cref="AssetPostprocessor.GetVersion"/> is left at 0 on purpose. A postprocessor with a
    /// version of its own becomes part of what every model and every texture import depends on, so
    /// installing or updating the package would reimport every model and texture in the project,
    /// most of which are not TerreinLab's. Models imported before the package was installed, or
    /// before an update, are reimported with <c>Tools/TerreinLab/Reimport models</c> instead.
    /// </para>
    /// </remarks>
    public sealed class TerreinLabPostprocessor : AssetPostprocessor
    {
        // Unity makes a postprocessor for each asset it imports; these remember, for the asset
        // this one is importing, whether it is a TerreinLab model. The user-properties callback
        // runs once per object, ten thousand times for a model full of trees, and should not read
        // metadata.json every time.
        string checkedPath;
        TerreinLabMetadata checkedMetadata;

        bool IsTerreinLabModel(out TerreinLabMetadata metadata)
        {
            if (!string.Equals(checkedPath, assetPath, StringComparison.Ordinal))
            {
                TerreinLabFolder.IsModel(assetPath, out checkedMetadata);
                checkedPath = assetPath;
            }
            metadata = checkedMetadata;
            return metadata != null;
        }

        /// <summary>Sets the model settings before Unity reads a TerreinLab FBX.</summary>
        void OnPreprocessModel()
        {
            if (!IsTerreinLabModel(out _))
                return;
            if (assetImporter is ModelImporter importer)
                TerreinLabImportSettings.ApplyModelSettings(importer);
        }

        /// <summary>Sets the texture settings for a texture lying directly in a TerreinLab folder.</summary>
        void OnPreprocessTexture()
        {
            if (!TerreinLabFolder.IsInTerreinLabFolder(assetPath))
                return;
            if (assetImporter is TextureImporter importer)
                TerreinLabImportSettings.ApplyTextureSettings(importer, assetPath);
        }

        /// <summary>
        /// Turns on GPU instancing on every material a TerreinLab model makes.
        /// </summary>
        /// <remarks>
        /// <c>M_tree</c> is the one that counts: ten thousand trees share twelve meshes and that one
        /// material, and with instancing each shape is drawn in one batch instead of one call per
        /// tree. The flag costs nothing on a material whose objects never repeat, and cars, lamp
        /// posts and bollards repeat too, so every material gets it. Materials extracted from the
        /// model are separate assets that never pass through here; the Reimport models menu item
        /// turns it on for those.
        /// </remarks>
        /// <param name="material">A material the model import made.</param>
        void OnPostprocessMaterial(Material material)
        {
            if (material == null || !IsTerreinLabModel(out _))
                return;
            material.enableInstancing = true;
        }

        /// <summary>
        /// Keeps the FBX custom properties of one object on a <see cref="TerreinLabProperties"/>
        /// component, where scripts and the Inspector can read them.
        /// </summary>
        /// <param name="gameObject">The object that carried user properties in the FBX.</param>
        /// <param name="propNames">The property names.</param>
        /// <param name="values">The values, in the same order.</param>
        void OnPostprocessGameObjectWithUserProperties(GameObject gameObject, string[] propNames, object[] values)
        {
            if (gameObject == null || propNames == null || values == null || !IsTerreinLabModel(out _))
                return;
            var properties = gameObject.GetComponent<TerreinLabProperties>();
            // Compared with == rather than ??, which would not see Unity's null for a missing component.
            if (properties == null)
                properties = gameObject.AddComponent<TerreinLabProperties>();
            int count = Math.Min(propNames.Length, values.Length);
            for (int i = 0; i < count; i++)
                properties.Set(propNames[i], values[i]);
        }

        /// <summary>
        /// Puts a <see cref="TerreinLabModel"/> on the root of a TerreinLab model, filled from the
        /// <c>metadata.json</c> beside it.
        /// </summary>
        /// <param name="root">The root of the imported model.</param>
        void OnPostprocessModel(GameObject root)
        {
            if (root == null || !IsTerreinLabModel(out TerreinLabMetadata metadata))
                return;

            var model = root.GetComponent<TerreinLabModel>();
            if (model == null)
                model = root.AddComponent<TerreinLabModel>();
            model.Configure(
                metadata.Name,
                metadata.OriginRdX,
                metadata.OriginRdY,
                metadata.GroundZOffsetNap,
                metadata.BboxRdXMin,
                metadata.BboxRdYMin,
                metadata.BboxRdXMax,
                metadata.BboxRdYMax);

            if (!metadata.AxisAsExpected)
            {
                Debug.LogWarning(
                    assetPath + ": " + TerreinLabNames.MetadataFile + " gives the axes as \"" + metadata.UnityAxis
                    + "\", not \"" + TerreinLabNames.UnityAxis + "\". TerreinLabModel converts to RD and NAP "
                    + "assuming the latter; check positions against the map before relying on them.",
                    root);
            }
        }

        /// <summary>
        /// Reimports the models and textures of a folder whose <c>metadata.json</c> or
        /// <c>ATTRIBUTION.txt</c> arrived or changed after they were imported.
        /// </summary>
        /// <remarks>
        /// Copying an output folder into a project while Unity has focus can import the FBX before
        /// the files that mark it as TerreinLab's have landed, and it would keep the settings it got
        /// then. Assets imported in the same batch as the marker files saw them already and are not
        /// imported twice. The reimport waits until this import has finished.
        /// </remarks>
        static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets, string[] movedAssets, string[] movedFromAssetPaths)
        {
            HashSet<string> imported = null;
            HashSet<string> folders = null;
            foreach (string path in importedAssets)
            {
                string file = TerreinLabFolder.FileNameOf(path);
                if (!string.Equals(file, TerreinLabNames.MetadataFile, StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(file, TerreinLabNames.AttributionFile, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (folders == null)
                    folders = new HashSet<string>(StringComparer.Ordinal);
                folders.Add(TerreinLabFolder.FolderOf(path));
            }
            if (folders == null)
                return;

            var stale = new List<string>();
            foreach (string folder in folders)
            {
                if (!TerreinLabFolder.TryRead(folder, out _))
                    continue;
                foreach (string asset in TerreinLabFolder.AssetPathsIn(folder))
                {
                    if (imported == null)
                        imported = new HashSet<string>(importedAssets, StringComparer.OrdinalIgnoreCase);
                    if (imported.Contains(asset) || !TerreinLabReimport.IsHandled(asset))
                        continue;
                    stale.Add(asset);
                }
            }
            if (stale.Count > 0)
                EditorApplication.delayCall += () => TerreinLabReimport.Reimport(stale, writeSettings: false);
        }
    }
}
