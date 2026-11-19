using System;
using System.IO;

namespace TerreinLab.Importing
{
    /// <summary>
    /// The size and alpha of a PNG, read from its header without decoding the image.
    /// </summary>
    /// <remarks>
    /// The importer has to choose the aerial's Max Size before Unity has read the image, and the
    /// first 26 bytes of a PNG say how big it is. An aerial at High detail over a few kilometres is
    /// several hundred megabytes decoded, so this is also far cheaper than loading it.
    /// </remarks>
    internal readonly struct PngInfo
    {
        static readonly byte[] Signature = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };

        /// <summary>Creates the header information of a PNG.</summary>
        /// <param name="width">Width in pixels.</param>
        /// <param name="height">Height in pixels.</param>
        /// <param name="hasAlpha">Whether the image has an alpha channel.</param>
        public PngInfo(int width, int height, bool hasAlpha)
        {
            Width = width;
            Height = height;
            HasAlpha = hasAlpha;
        }

        /// <summary>Width in pixels.</summary>
        public int Width { get; }

        /// <summary>Height in pixels.</summary>
        public int Height { get; }

        /// <summary>The longer side in pixels, which is what Max Size limits.</summary>
        public int LargestSide => Math.Max(Width, Height);

        /// <summary>
        /// Whether the image has an alpha channel: colour type 4 (grey and alpha) or 6 (RGBA), as
        /// Pillow writes <c>fence.png</c>.
        /// </summary>
        public bool HasAlpha { get; }

        /// <summary>Reads the header of a PNG file.</summary>
        /// <param name="path">The file, as a path relative to the project folder or absolute.</param>
        /// <param name="info">The size and alpha, or all zero when the file is not a readable PNG.</param>
        /// <returns>True when the file starts with a valid PNG header.</returns>
        public static bool TryRead(string path, out PngInfo info)
        {
            info = default;
            var header = new byte[26];
            try
            {
                using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                {
                    int read = 0;
                    while (read < header.Length)
                    {
                        int count = stream.Read(header, read, header.Length - read);
                        if (count <= 0)
                            return false;
                        read += count;
                    }
                }
            }
            catch (IOException)
            {
                return false;
            }
            catch (UnauthorizedAccessException)
            {
                return false;
            }
            return TryParse(header, out info);
        }

        /// <summary>Parses the first 26 bytes of a PNG: the signature and the IHDR chunk up to the colour type.</summary>
        /// <param name="header">At least 26 bytes from the start of the file.</param>
        /// <param name="info">The size and alpha, or all zero when this is not a PNG header.</param>
        public static bool TryParse(byte[] header, out PngInfo info)
        {
            info = default;
            if (header == null || header.Length < 26)
                return false;
            for (int i = 0; i < Signature.Length; i++)
            {
                if (header[i] != Signature[i])
                    return false;
            }
            // Bytes 8-11 are the chunk length, 12-15 its type; IHDR always comes first.
            if (header[12] != (byte)'I' || header[13] != (byte)'H' || header[14] != (byte)'D' || header[15] != (byte)'R')
                return false;
            long width = ReadBigEndian(header, 16);
            long height = ReadBigEndian(header, 20);
            if (width <= 0 || height <= 0 || width > int.MaxValue || height > int.MaxValue)
                return false;
            byte colourType = header[25];
            info = new PngInfo((int)width, (int)height, colourType == 4 || colourType == 6);
            return true;
        }

        static long ReadBigEndian(byte[] bytes, int offset)
        {
            return ((long)bytes[offset] << 24)
                | ((long)bytes[offset + 1] << 16)
                | ((long)bytes[offset + 2] << 8)
                | bytes[offset + 3];
        }
    }
}
