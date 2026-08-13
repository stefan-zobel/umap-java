package tagbio.umap.fx;

/**
 * The parameter values the viewer is allowed to offer.
 *
 * <p>These are not cosmetic limits. The Java port replaces scipy's <code>curve_fit</code>
 * with interpolation in a fixed lookup table, and values outside that table either fail the
 * argument check or index past the end of a table row.
 */
public final class ParameterRange {

  /** Smallest spread the lookup table covers. */
  public static final double MIN_SPREAD = 0.5;

  /** Largest spread the lookup table covers. */
  public static final double MAX_SPREAD = 1.5;

  /** Increment used by both the spread and the minimum distance slider. */
  public static final double STEP = 0.05;

  private ParameterRange() { }

  /**
   * The largest minimum distance that can be combined with a given spread.
   *
   * <p>The lookup table row for a spread is selected by <code>(int) (10 * spread)</code>
   * and holds <code>2 * (index + 1)</code> entries, while the interpolation also reads the
   * entry after the one it lands on. The distance index is <code>(int) (20 * minDist)</code>,
   * so it has to stay at or below <code>2 * index</code>. Rounding the spread down to a
   * tenth satisfies that for every spread, and is exact when the spread is itself a multiple
   * of a tenth. The index is computed in float so that it matches what the curve fit will
   * see when the value is passed on.
   *
   * @param spread the spread the projection will use
   * @return the inclusive upper bound for the minimum distance
   */
  public static double maxMinDist(final double spread) {
    final int spreadIndex = (int) (10 * (float) spread);
    return spreadIndex / 10.0;
  }
}
