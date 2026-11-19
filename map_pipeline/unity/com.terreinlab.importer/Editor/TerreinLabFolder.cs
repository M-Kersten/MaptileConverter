using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TerreinLab.Importing
{
    /// <summary>
    /// Recognises a TerreinLab output folder and the model in it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A folder is a TerreinLab folder when it holds both <c>metadata.json</c> and
    /// <c>ATTRIBUTION.txt</c> and the metadata carries the RD origin (see
    /// <see cref="TerreinLabMetadata"/>). A model is an <c>.fbx</c> directly in such a folder;
    /// a texture is processed when it lies directly in one. Subfolders such as <c>preview/</c> are
    /// not part of it, and every other asset in the project is left alone.
    /// </para>
    /// <para>
    /// Paths are Unity asset paths relative to the project folder, such as
    /// <c>Assets/Models/utrecht_dom/model.fbx</c>; the files are read straight from disk, which is
    /// allowed during an import where the asset database is not.
    /// </para>
    /// </remarks>
    internal static class TerreinLabFolder
    {
        /// <summary>The folder part of an asset path, with forward slashes and no trailing slash.</summary>
        /// <param name="assetPath">An asset path such as <c>Assets/area/model.fbx</c>.</param>
        public static string FolderOf(string assetPath)
        {
            if (string.IsNullOrEmpty(assetPath))
                return string.Empty;
            string path = assetPath.Replace('\\', '/');
            int slash = path.LastIndexOf('/');
            return slash < 0 ? string.Empty : path.Substring(0, slash);
        }

        /// <summary>The file name part of an asset path.</summary>
        /// <param name="assetPath">An asset path such as <c>Assets/area/aerial.png</c>.</param>
        public static string FileNameOf(string assetPath)
        {
            if (string.IsNullOrEmpty(assetPath))
                return string.Empty;
            string path = assetPath.Replace('\\', '/');
            return path.Substring(path.LastIndexOf('/') + 1);
        }

        /// <summary>A file in a folder, as an asset path.</summary>
        /// <param name="folder">An asset folder such as <c>Assets/area</c>, or empty for the project folder.</param>
        /// <param name="fileName">The file name.</param>
        public static string Combine(string folder, string fileName)
        {
            return string.IsNullOrEmpty(folder) ? fileName : folder + "/" + fileName;
        }

        /// <summary>Reads the metadata of a folder when it is a TerreinLab folder.</summary>
        /// <param name="folder">An asset folder such as <c>Assets/area</c>.</param>
        /// <param name="metadata">The metadata, or null.</param>
        /// <param name="reason">Why the folder is not a TerreinLab folder, or null.</param>
        /// <returns>True when the folder is a TerreinLab folder.</returns>
        public static bool TryRead(string folder, out TerreinLabMetadata metadata, out string reason)
        {
            metadata = null;
            string metadataPath = Combine(folder, TerreinLabNames.MetadataFile);
            if (!File.Exists(metadataPath))
            {
                reason = "no " + TerreinLabNames.MetadataFile;
                return false;
            }
            if (!File.Exists(Combine(folder, TerreinLabNames.AttributionFile)))
            {
                reason = "no " + TerreinLabNames.AttributionFile;
                return false;
            }

            string json;
            try
            {
                json = File.ReadAllText(metadataPath, Encoding.UTF8);
            }
            catch (IOException exception)
            {
                reason = "cannot read " + TerreinLabNames.MetadataFile + ": " + exception.Message;
                return false;
            }
            catch (UnauthorizedAccessException exception)
            {
                reason = "cannot read " + TerreinLabNames.MetadataFile + ": " + exception.Message;
                return false;
            }

            if (!TerreinLabMetadata.TryParse(json, out metadata, out string why))
            {
                reason = TerreinLabNames.MetadataFile + " is " + why;
                return false;
            }
            reason = null;
            return true;
        }

        /// <summary>Reads the metadata of a folder when it is a TerreinLab folder.</summary>
        /// <param name="folder">An asset folder such as <c>Assets/area</c>.</param>
        /// <param name="metadata">The metadata, or null.</param>
        public static bool TryRead(string folder, out TerreinLabMetadata metadata)
        {
            return TryRead(folder, out metadata, out _);
        }

        /// <summary>Whether an asset is a TerreinLab model: an <c>.fbx</c> in a TerreinLab folder.</summary>
        /// <param name="assetPath">The asset path.</param>
        /// <param name="metadata">The metadata beside it, or null.</param>
        public static bool IsModel(string assetPath, out TerreinLabMetadata metadata)
        {
            metadata = null;
            if (string.IsNullOrEmpty(assetPath)
                || !assetPath.EndsWith(TerreinLabNames.ModelExtension, StringComparison.OrdinalIgnoreCase))
                return false;
            return TryRead(FolderOf(assetPath), out metadata);
        }

        /// <summary>Whether an asset lies directly in a TerreinLab folder.</summary>
        /// <param name="assetPath">The asset path.</param>
        public static bool IsInTerreinLabFolder(string assetPath)
        {
            return !string.IsNullOrEmpty(assetPath) && TryRead(FolderOf(assetPath), out _);
        }

        /// <summary>
        /// The files directly in a folder as asset paths, sorted, without <c>.meta</c> files.
        /// Empty when the folder cannot be listed.
        /// </summary>
        /// <param name="folder">An asset folder such as <c>Assets/area</c>.</param>
        public static List<string> AssetPathsIn(string folder)
        {
            var result = new List<string>();
            string[] files;
            try
            {
                files = Directory.GetFiles(string.IsNullOrEmpty(folder) ? "." : folder);
            }
            catch (IOException)
            {
                return result;
            }
            catch (UnauthorizedAccessException)
            {
                return result;
            }

            foreach (string file in files)
            {
                if (file.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                    continue;
                result.Add(Combine(folder, FileNameOf(file)));
            }
            result.Sort(StringComparer.Ordinal);
            return result;
        }
    }
}
