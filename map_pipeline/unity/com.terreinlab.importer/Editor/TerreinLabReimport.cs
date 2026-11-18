using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace TerreinLab.Importing
{
    /// <summary>
    /// The <c>Tools/TerreinLab/Reimport models</c> menu item, and the reimport it shares with the
    /// postprocessor.
    /// </summary>
    internal static class TerreinLabReimport
    {
        /// <summary>Where the menu item sits.</summary>
        public const string MenuPath = "Tools/TerreinLab/Reimport models";

        /// <summary>
        /// Reimports every TerreinLab model in the project and the textures beside it, for models
        /// imported before this package was installed or updated.
        /// </summary>
        /// <remarks>
        /// The settings are also written into each asset's import settings first, so the Inspector
        /// shows what the import uses. Materials extracted from a model are separate assets the
        /// import does not touch, so GPU instancing is turned on in those here.
        /// </remarks>
        [MenuItem(MenuPath)]
        static void ReimportModels()
        {
            var models = new List<string>();
            var folders = new SortedSet<string>(StringComparer.Ordinal);
            foreach (string path in AssetDatabase.GetAllAssetPaths())
            {
                if (!TerreinLabFolder.IsModel(path, out _))
                    continue;
                models.Add(path);
                folders.Add(TerreinLabFolder.FolderOf(path));
            }

            if (models.Count == 0)
            {
                EditorUtility.DisplayDialog(
                    "TerreinLab",
                    "No TerreinLab models found. A TerreinLab model is an .fbx with "
                    + TerreinLabNames.MetadataFile + " and " + TerreinLabNames.AttributionFile
                    + " beside it, as the pipeline writes them.",
                    "OK");
                return;
            }

            int materials = EnableInstancingOnExtractedMaterials(models, out bool treeMaterial);

            var assets = new List<string>();
            foreach (string folder in folders)
            {
                foreach (string asset in TerreinLabFolder.AssetPathsIn(folder))
                {
                    if (IsHandled(asset))
                        assets.Add(asset);
                }
            }
            Reimport(assets, writeSettings: true);

            string message = string.Format(
                "TerreinLab: reimported {0} model(s) and {1} texture(s) in {2} folder(s).",
                models.Count,
                assets.Count - models.Count,
                folders.Count);
            if (materials > 0)
            {
                message += string.Format(
                    " Turned GPU instancing on in {0} extracted material(s){1}.",
                    materials,
                    treeMaterial ? ", " + TerreinLabNames.TreeMaterial + " among them" : string.Empty);
            }
            Debug.Log(message);
        }

        /// <summary>Whether an asset in a TerreinLab folder is one the importer sets up: the model or a PNG.</summary>
        /// <param name="assetPath">The asset path.</param>
        public static bool IsHandled(string assetPath)
        {
            return assetPath.EndsWith(TerreinLabNames.ModelExtension, StringComparison.OrdinalIgnoreCase)
                || assetPath.EndsWith(".png", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Reimports assets in TerreinLab folders in one batch, textures before models.
        /// </summary>
        /// <param name="assets">Asset paths of models and textures.</param>
        /// <param name="writeSettings">
        /// Also write the TerreinLab settings into each asset's import settings first. The import
        /// applies them either way; this makes the <c>.meta</c> files and the Inspector agree.
        /// </param>
        public static void Reimport(IList<string> assets, bool writeSettings)
        {
            if (assets == null || assets.Count == 0)
                return;

            var ordered = new List<string>(assets.Count);
            foreach (string asset in assets)
            {
                if (!asset.EndsWith(TerreinLabNames.ModelExtension, StringComparison.OrdinalIgnoreCase))
                    ordered.Add(asset);
            }
            foreach (string asset in assets)
            {
                if (asset.EndsWith(TerreinLabNames.ModelExtension, StringComparison.OrdinalIgnoreCase))
                    ordered.Add(asset);
            }

            AssetDatabase.StartAssetEditing();
            try
            {
                foreach (string asset in ordered)
                {
                    if (writeSettings)
                        WriteSettings(asset);
                    AssetDatabase.ImportAsset(asset, ImportAssetOptions.ForceUpdate);
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
            }
        }

        static void WriteSettings(string asset)
        {
            AssetImporter importer = AssetImporter.GetAtPath(asset);
            if (importer is ModelImporter model && TerreinLabFolder.IsModel(asset, out _))
                TerreinLabImportSettings.ApplyModelSettings(model);
            else if (importer is TextureImporter texture && TerreinLabFolder.IsInTerreinLabFolder(asset))
                TerreinLabImportSettings.ApplyTextureSettings(texture, asset);
            else
                return;
            AssetDatabase.WriteImportSettingsIfDirty(asset);
        }

        static int EnableInstancingOnExtractedMaterials(List<string> models, out bool treeMaterial)
        {
            treeMaterial = false;
            int changed = 0;
            foreach (string path in models)
            {
                if (!(AssetImporter.GetAtPath(path) is ModelImporter importer))
                    continue;
                foreach (var pair in importer.GetExternalObjectMap())
                {
                    // Only .mat files: a material remapped to one inside another model is that
                    // model's to set.
                    if (!(pair.Value is Material material) || material.enableInstancing
                        || !AssetDatabase.IsNativeAsset(material))
                        continue;
                    material.enableInstancing = true;
                    EditorUtility.SetDirty(material);
                    changed++;
                    if (material.name == TerreinLabNames.TreeMaterial)
                        treeMaterial = true;
                }
            }
            if (changed > 0)
                AssetDatabase.SaveAssets();
            return changed;
        }
    }
}
