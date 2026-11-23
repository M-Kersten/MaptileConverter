using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace TerreinLab.Importing
{
    /// <summary>
    /// A small JSON reader for <c>metadata.json</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>JsonUtility</c> only fills the fields of a class declared up front, cannot tell a missing
    /// number from a zero, and rejects the <c>NaN</c> and <c>Infinity</c> that Python's
    /// <c>json.dumps</c> writes for a statistic that could not be computed. The metadata is a
    /// handful of kilobytes of nested objects and arrays of which only a few top-level fields
    /// matter, so it is read whole into plain objects instead.
    /// </para>
    /// <para>
    /// Objects become <see cref="Dictionary{TKey,TValue}"/> of string to object, arrays
    /// <see cref="List{T}"/> of object, numbers <see cref="double"/>, and true, false and null
    /// <see cref="bool"/> and null.
    /// </para>
    /// </remarks>
    internal static class MiniJson
    {
        const int MaxDepth = 128;

        /// <summary>Parses a whole JSON document.</summary>
        /// <param name="text">The document.</param>
        /// <returns>The value it holds.</returns>
        /// <exception cref="FormatException">The text is not JSON.</exception>
        public static object Parse(string text)
        {
            if (text == null)
                throw new ArgumentNullException(nameof(text));
            var reader = new Reader(text);
            reader.SkipWhitespace();
            object value = reader.ReadValue(0);
            reader.SkipWhitespace();
            if (!reader.AtEnd)
                throw reader.Error("unexpected text after the end of the document");
            return value;
        }

        /// <summary>Parses a JSON document, or returns false with the reason it is not JSON.</summary>
        /// <param name="text">The document.</param>
        /// <param name="value">The value it holds, or null.</param>
        /// <param name="error">Why it could not be read, or null.</param>
        public static bool TryParse(string text, out object value, out string error)
        {
            try
            {
                value = Parse(text);
                error = null;
                return true;
            }
            catch (FormatException exception)
            {
                value = null;
                error = exception.Message;
                return false;
            }
            catch (ArgumentNullException)
            {
                value = null;
                error = "no text";
                return false;
            }
        }

        sealed class Reader
        {
            readonly string text;
            int position;

            public Reader(string text)
            {
                this.text = text;
                // A byte order mark is not JSON, but an editor may well have saved one.
                position = text.Length > 0 && text[0] == '﻿' ? 1 : 0;
            }

            public bool AtEnd => position >= text.Length;

            public FormatException Error(string message)
            {
                return new FormatException(
                    string.Format(CultureInfo.InvariantCulture, "{0} at character {1}", message, position));
            }

            public void SkipWhitespace()
            {
                while (position < text.Length)
                {
                    char c = text[position];
                    if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
                        return;
                    position++;
                }
            }

            public object ReadValue(int depth)
            {
                if (depth > MaxDepth)
                    throw Error("nested too deeply");
                if (AtEnd)
                    throw Error("unexpected end of the document");

                char c = text[position];
                switch (c)
                {
                    case '{':
                        return ReadObject(depth);
                    case '[':
                        return ReadArray(depth);
                    case '"':
                        return ReadString();
                    case 't':
                        Expect("true");
                        return true;
                    case 'f':
                        Expect("false");
                        return false;
                    case 'n':
                        Expect("null");
                        return null;
                    case 'N':
                        Expect("NaN");
                        return double.NaN;
                    case 'I':
                        Expect("Infinity");
                        return double.PositiveInfinity;
                    default:
                        if (c == '-' || (c >= '0' && c <= '9'))
                            return ReadNumber();
                        throw Error("unexpected character '" + c + "'");
                }
            }

            Dictionary<string, object> ReadObject(int depth)
            {
                var result = new Dictionary<string, object>(StringComparer.Ordinal);
                position++; // {
                SkipWhitespace();
                if (TryConsume('}'))
                    return result;
                while (true)
                {
                    SkipWhitespace();
                    if (AtEnd || text[position] != '"')
                        throw Error("expected a quoted key");
                    string key = ReadString();
                    SkipWhitespace();
                    if (!TryConsume(':'))
                        throw Error("expected ':' after a key");
                    SkipWhitespace();
                    // The last of a repeated key wins, as in Python.
                    result[key] = ReadValue(depth + 1);
                    SkipWhitespace();
                    if (TryConsume(','))
                        continue;
                    if (TryConsume('}'))
                        return result;
                    throw Error("expected ',' or '}' in an object");
                }
            }

            List<object> ReadArray(int depth)
            {
                var result = new List<object>();
                position++; // [
                SkipWhitespace();
                if (TryConsume(']'))
                    return result;
                while (true)
                {
                    SkipWhitespace();
                    result.Add(ReadValue(depth + 1));
                    SkipWhitespace();
                    if (TryConsume(','))
                        continue;
                    if (TryConsume(']'))
                        return result;
                    throw Error("expected ',' or ']' in an array");
                }
            }

            string ReadString()
            {
                position++; // opening quote
                var builder = new StringBuilder();
                while (true)
                {
                    if (AtEnd)
                        throw Error("unterminated string");
                    char c = text[position++];
                    if (c == '"')
                        return builder.ToString();
                    if (c != '\\')
                    {
                        builder.Append(c);
                        continue;
                    }
                    if (AtEnd)
                        throw Error("unterminated escape");
                    char escaped = text[position++];
                    switch (escaped)
                    {
                        case '"': builder.Append('"'); break;
                        case '\\': builder.Append('\\'); break;
                        case '/': builder.Append('/'); break;
                        case 'b': builder.Append('\b'); break;
                        case 'f': builder.Append('\f'); break;
                        case 'n': builder.Append('\n'); break;
                        case 'r': builder.Append('\r'); break;
                        case 't': builder.Append('\t'); break;
                        case 'u':
                            if (position + 4 > text.Length)
                                throw Error("short \\u escape");
                            if (!int.TryParse(
                                    text.Substring(position, 4),
                                    NumberStyles.AllowHexSpecifier,
                                    CultureInfo.InvariantCulture,
                                    out int code))
                                throw Error("bad \\u escape");
                            // Surrogate pairs arrive as two escapes and are appended one half
                            // at a time, which reassembles them.
                            builder.Append((char)code);
                            position += 4;
                            break;
                        default:
                            throw Error("unknown escape '\\" + escaped + "'");
                    }
                }
            }

            object ReadNumber()
            {
                int start = position;
                if (text[position] == '-')
                {
                    position++;
                    if (!AtEnd && text[position] == 'I')
                    {
                        Expect("Infinity");
                        return double.NegativeInfinity;
                    }
                }
                while (position < text.Length)
                {
                    char c = text[position];
                    if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
                        position++;
                    else
                        break;
                }
                string token = text.Substring(start, position - start);
                if (!double.TryParse(
                        token,
                        NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent,
                        CultureInfo.InvariantCulture,
                        out double number))
                {
                    position = start;
                    throw Error("bad number '" + token + "'");
                }
                return number;
            }

            void Expect(string word)
            {
                if (string.CompareOrdinal(text, position, word, 0, word.Length) != 0)
                    throw Error("expected '" + word + "'");
                position += word.Length;
            }

            bool TryConsume(char c)
            {
                if (position < text.Length && text[position] == c)
                {
                    position++;
                    return true;
                }
                return false;
            }
        }
    }
}
