using UnityEditor;
using UnityEngine;

namespace TerreinLab.Importing
{
    /// <summary>
    /// Shows a <see cref="TerreinLabProperties"/> as one line per property, name and value, instead
    /// of two separate lists. The values can be selected and copied, a BAG id for instance.
    /// </summary>
    [CustomEditor(typeof(TerreinLabProperties))]
    internal sealed class TerreinLabPropertiesEditor : UnityEditor.Editor
    {
        /// <summary>Draws the properties.</summary>
        public override void OnInspectorGUI()
        {
            var properties = (TerreinLabProperties)target;
            int count = properties.Count;
            if (count == 0)
            {
                EditorGUILayout.HelpBox("No properties.", MessageType.None);
                return;
            }
            for (int i = 0; i < count; i++)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.PrefixLabel(properties.Keys[i]);
                EditorGUILayout.SelectableLabel(
                    properties.Values[i], GUILayout.Height(EditorGUIUtility.singleLineHeight));
                EditorGUILayout.EndHorizontal();
            }
        }
    }
}
