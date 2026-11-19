using System;
using System.Globalization;

namespace TerreinLab
{
    /// <summary>
    /// A position in the Dutch national grid: RD New (EPSG:28992) easting and northing in metres,
    /// and the height above NAP in metres.
    /// </summary>
    /// <remarks>
    /// Doubles throughout, because an RD coordinate is six figures before the decimal point and a
    /// float keeps only about seven significant figures: that would round every position to the
    /// nearest few centimetres. Get one from <see cref="TerreinLabModel.WorldToRdNap"/>.
    /// </remarks>
    public readonly struct RdNapPosition : IEquatable<RdNapPosition>
    {
        /// <summary>Creates a position from RD easting, RD northing and NAP height.</summary>
        /// <param name="rdX">RD easting in metres.</param>
        /// <param name="rdY">RD northing in metres.</param>
        /// <param name="nap">Height above NAP in metres. Negative below NAP, which is common.</param>
        public RdNapPosition(double rdX, double rdY, double nap)
        {
            RdX = rdX;
            RdY = rdY;
            Nap = nap;
        }

        /// <summary>RD easting in metres (<c>rd_x</c> in <c>metadata.json</c>).</summary>
        public double RdX { get; }

        /// <summary>RD northing in metres (<c>rd_y</c> in <c>metadata.json</c>).</summary>
        public double RdY { get; }

        /// <summary>Height above NAP in metres.</summary>
        public double Nap { get; }

        /// <inheritdoc />
        public bool Equals(RdNapPosition other)
        {
            return RdX.Equals(other.RdX) && RdY.Equals(other.RdY) && Nap.Equals(other.Nap);
        }

        /// <inheritdoc />
        public override bool Equals(object obj)
        {
            return obj is RdNapPosition other && Equals(other);
        }

        /// <inheritdoc />
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = RdX.GetHashCode();
                hash = (hash * 397) ^ RdY.GetHashCode();
                return (hash * 397) ^ Nap.GetHashCode();
            }
        }

        /// <summary>The position as "RD 136900.000, 455850.000; NAP 5.186 m", to the millimetre.</summary>
        public override string ToString()
        {
            return string.Format(
                CultureInfo.InvariantCulture, "RD {0:F3}, {1:F3}; NAP {2:F3} m", RdX, RdY, Nap);
        }

        /// <summary>True when both positions are exactly the same.</summary>
        public static bool operator ==(RdNapPosition left, RdNapPosition right)
        {
            return left.Equals(right);
        }

        /// <summary>True when the positions differ in any coordinate.</summary>
        public static bool operator !=(RdNapPosition left, RdNapPosition right)
        {
            return !left.Equals(right);
        }
    }
}
