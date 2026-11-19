using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

namespace TerreinLab
{
    /// <summary>
    /// The custom properties one object of a TerreinLab model carries, as Blender wrote them into
    /// the FBX: <c>bag_id</c>, <c>build_year</c> and <c>height_m</c> on a building, <c>kind</c> on a
    /// road, <c>height_m</c> and <c>detected</c> on a tree, <c>maptile_role</c> and
    /// <c>maptile_ground</c> on the pieces of ground.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The TerreinLab importer adds this to every object of the model that has FBX user properties.
    /// Keys and values are kept as two parallel lists of text, the values written in the invariant
    /// culture, so whatever the FBX held survives; the typed accessors read them back.
    /// </para>
    /// <para>
    /// A <c>true</c> written in Blender arrives through FBX as the integer 1, so
    /// <see cref="TryGetBool"/> reads 1 and 0 as well as true and false.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var properties = hit.collider.GetComponent&lt;TerreinLabProperties&gt;();
    /// if (properties != null &amp;&amp; properties.BagId != null)
    ///     Debug.Log($"{properties.BagId}, built {properties.BuildYear}, {properties.HeightM} m");
    /// </code>
    /// </example>
    [DisallowMultipleComponent]
    [AddComponentMenu("TerreinLab/TerreinLab Properties")]
    public sealed class TerreinLabProperties : MonoBehaviour
    {
        /// <summary>The BAG building identifier, such as <c>NL.IMBAG.Pand.0344100000031316</c>.</summary>
        public const string BagIdKey = "bag_id";

        /// <summary>The original construction year of a building, from 3DBAG.</summary>
        public const string BuildYearKey = "build_year";

        /// <summary>The height of a building or a tree in metres.</summary>
        public const string HeightMKey = "height_m";

        /// <summary>What a road object is: <c>street</c>, <c>junction</c> or <c>paving</c>.</summary>
        public const string KindKey = "kind";

        /// <summary>Whether a tree was found in the height model rather than taken from the BGT register.</summary>
        public const string DetectedKey = "detected";

        /// <summary>
        /// What a piece of ground is: <c>terrain</c>, <c>road</c>, <c>walkway</c>, <c>grass</c>,
        /// <c>water</c>, or <c>harbour_</c> followed by the kind of harbour work.
        /// </summary>
        public const string MaptileRoleKey = "maptile_role";

        /// <summary>Set on every piece of the ground sheet in the layered style.</summary>
        public const string MaptileGroundKey = "maptile_ground";

        [SerializeField]
        [Tooltip("Property names, in the order the FBX listed them.")]
        List<string> keys = new List<string>();

        [SerializeField]
        [Tooltip("Property values as text, invariant culture, in the same order as the names.")]
        List<string> values = new List<string>();

        /// <summary>How many properties there are.</summary>
        public int Count => Math.Min(keys.Count, values.Count);

        /// <summary>The property names, in the order the FBX listed them.</summary>
        public IReadOnlyList<string> Keys => keys;

        /// <summary>The property values as text, in the same order as <see cref="Keys"/>.</summary>
        public IReadOnlyList<string> Values => values;

        /// <summary>The BAG building identifier, or null when this is not a building.</summary>
        public string BagId => GetString(BagIdKey);

        /// <summary>The construction year of a building, or null when there is none.</summary>
        public int? BuildYear => TryGetInt(BuildYearKey, out int year) ? year : (int?)null;

        /// <summary>The height in metres of a building or tree, or null when there is none.</summary>
        public float? HeightM => TryGetFloat(HeightMKey, out float height) ? height : (float?)null;

        /// <summary>What a road object is (<c>street</c>, <c>junction</c>, <c>paving</c>), or null.</summary>
        public string Kind => GetString(KindKey);

        /// <summary>
        /// For a tree, true when it was found in the height model and false when it is in the BGT
        /// register; null on anything that is not a tree.
        /// </summary>
        public bool? Detected => TryGetBool(DetectedKey, out bool detected) ? detected : (bool?)null;

        /// <summary>What a piece of ground is (<c>terrain</c>, <c>road</c>, <c>water</c> and so on), or null.</summary>
        public string MaptileRole => GetString(MaptileRoleKey);

        /// <summary>True on the pieces of the layered style's ground sheet.</summary>
        public bool IsGround => TryGetBool(MaptileGroundKey, out bool ground) && ground;

        /// <summary>Whether a property of this name is present.</summary>
        /// <param name="key">The property name, such as <c>bag_id</c>. Case-sensitive.</param>
        public bool Has(string key)
        {
            return IndexOf(key) >= 0;
        }

        /// <summary>Reads a property as text.</summary>
        /// <param name="key">The property name. Case-sensitive.</param>
        /// <param name="value">The value as stored, or null when the property is absent.</param>
        /// <returns>True when the property is present.</returns>
        public bool TryGetString(string key, out string value)
        {
            int index = IndexOf(key);
            value = index >= 0 ? values[index] : null;
            return index >= 0;
        }

        /// <summary>Reads a property as text, or returns <paramref name="fallback"/> when it is absent.</summary>
        /// <param name="key">The property name. Case-sensitive.</param>
        /// <param name="fallback">What to return when the property is absent.</param>
        public string GetString(string key, string fallback = null)
        {
            return TryGetString(key, out string value) ? value : fallback;
        }

        /// <summary>Reads a property as a whole number.</summary>
        /// <param name="key">The property name. Case-sensitive.</param>
        /// <param name="value">The number, or 0 when the property is absent or not a whole number.</param>
        /// <returns>True when the property is present and a whole number.</returns>
        public bool TryGetInt(string key, out int value)
        {
            value = 0;
            if (!TryGetString(key, out string text))
                return false;
            if (int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
                return true;
            // A whole number stored as a float, such as "1890.0".
            if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number)
                && Math.Abs(number - Math.Round(number)) < 1e-9
                && number >= int.MinValue && number <= int.MaxValue)
            {
                value = (int)Math.Round(number);
                return true;
            }
            value = 0;
            return false;
        }

        /// <summary>Reads a property as a number.</summary>
        /// <param name="key">The property name. Case-sensitive.</param>
        /// <param name="value">The number, or 0 when the property is absent or not a number.</param>
        /// <returns>True when the property is present and a number.</returns>
        public bool TryGetFloat(string key, out float value)
        {
            value = 0f;
            return TryGetString(key, out string text)
                && float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        /// <summary>
        /// Reads a property as true or false: <c>true</c> and <c>false</c> in any case, or a number,
        /// where anything but zero is true.
        /// </summary>
        /// <param name="key">The property name. Case-sensitive.</param>
        /// <param name="value">The value, or false when the property is absent or neither.</param>
        /// <returns>True when the property is present and reads as true or false.</returns>
        public bool TryGetBool(string key, out bool value)
        {
            value = false;
            if (!TryGetString(key, out string text))
                return false;
            if (bool.TryParse(text, out value))
                return true;
            if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
            {
                value = number != 0.0;
                return true;
            }
            value = false;
            return false;
        }

        /// <summary>
        /// Sets a property, replacing one of the same name. Numbers are written in the invariant
        /// culture so that they read back the same on every machine. The importer calls this.
        /// </summary>
        /// <param name="key">The property name. Null or empty names are ignored.</param>
        /// <param name="value">
        /// The value, as Unity hands FBX user properties over: a string, bool, int, float, Color or
        /// Vector4. Anything else is stored as its invariant text.
        /// </param>
        public void Set(string key, object value)
        {
            if (string.IsNullOrEmpty(key))
                return;
            string text = Format(value);
            int index = IndexOf(key);
            if (index >= 0)
            {
                values[index] = text;
                return;
            }
            // Keep the two lists the same length even if a hand edit in the Inspector broke that.
            while (values.Count > keys.Count)
                values.RemoveAt(values.Count - 1);
            while (values.Count < keys.Count)
                values.Add(string.Empty);
            keys.Add(key);
            values.Add(text);
        }

        /// <summary>Removes every property.</summary>
        public void Clear()
        {
            keys.Clear();
            values.Clear();
        }

        /// <summary>
        /// Writes a property value as text in the invariant culture, floats so that they read back
        /// exactly. Exposed for the importer and for tests.
        /// </summary>
        /// <param name="value">The value to write.</param>
        public static string Format(object value)
        {
            switch (value)
            {
                case null:
                    return string.Empty;
                case string text:
                    return text;
                case bool flag:
                    return flag ? "true" : "false";
                case float single:
                    return single.ToString("R", CultureInfo.InvariantCulture);
                case double number:
                    return number.ToString("R", CultureInfo.InvariantCulture);
                case Color colour:
                    return Join(colour.r, colour.g, colour.b, colour.a);
                case Vector4 vector:
                    return Join(vector.x, vector.y, vector.z, vector.w);
                case IFormattable formattable:
                    return formattable.ToString(null, CultureInfo.InvariantCulture);
                default:
                    return value.ToString();
            }
        }

        static string Join(float a, float b, float c, float d)
        {
            return string.Join(
                ", ",
                a.ToString("R", CultureInfo.InvariantCulture),
                b.ToString("R", CultureInfo.InvariantCulture),
                c.ToString("R", CultureInfo.InvariantCulture),
                d.ToString("R", CultureInfo.InvariantCulture));
        }

        int IndexOf(string key)
        {
            if (key == null)
                return -1;
            int count = Count;
            for (int i = 0; i < count; i++)
            {
                if (string.Equals(keys[i], key, StringComparison.Ordinal))
                    return i;
            }
            return -1;
        }
    }
}
