package tagbio.umap;

import junit.framework.TestCase;

import tagbio.umap.fx.ParameterRange;

/**
 * Checks that every parameter combination the viewer's sliders can produce is one that the
 * curve fit lookup table actually covers. The argument check inside the curve fit is not
 * sufficient on its own: it permits spread and minimum distance pairs whose table row is
 * too short for the interpolation, which fails with an index out of bounds instead.
 */
public class ParameterRangeTest extends TestCase {

  private static int steps(final double from, final double to) {
    return (int) Math.round((to - from) / ParameterRange.STEP);
  }

  public void testEverySliderCombinationIsSupported() {
    final int spreadSteps = steps(ParameterRange.MIN_SPREAD, ParameterRange.MAX_SPREAD);
    assertEquals(20, spreadSteps);
    int combinations = 0;
    for (int i = 0; i <= spreadSteps; ++i) {
      final double spread = ParameterRange.MIN_SPREAD + i * ParameterRange.STEP;
      final double maxDist = ParameterRange.maxMinDist(spread);
      for (int j = 0; j <= steps(0, maxDist); ++j) {
        final double minDist = j * ParameterRange.STEP;
        final String at = "spread " + spread + ", minDist " + minDist;
        final float[] fit;
        try {
          fit = Curve.curveFit((float) spread, (float) minDist);
        } catch (final RuntimeException e) {
          fail(at + " threw " + e);
          return;
        }
        assertEquals(at, 2, fit.length);
        assertTrue(at, Float.isFinite(fit[0]));
        assertTrue(at, Float.isFinite(fit[1]));
        ++combinations;
      }
    }
    // Guard against the loops silently collapsing to nothing.
    assertTrue("only " + combinations + " combinations exercised", combinations > 200);
  }

  public void testMaxMinDistRoundsSpreadDownToATenth() {
    assertEquals(0.5, ParameterRange.maxMinDist(0.5), 1.0e-9);
    assertEquals(0.5, ParameterRange.maxMinDist(0.55), 1.0e-9);
    assertEquals(1.0, ParameterRange.maxMinDist(1.0), 1.0e-9);
    assertEquals(1.0, ParameterRange.maxMinDist(1.05), 1.0e-9);
    assertEquals(1.4, ParameterRange.maxMinDist(1.45), 1.0e-9);
    assertEquals(1.5, ParameterRange.maxMinDist(1.5), 1.0e-9);
  }

  public void testMaxMinDistNeverExceedsTheSpread() {
    final int spreadSteps = steps(ParameterRange.MIN_SPREAD, ParameterRange.MAX_SPREAD);
    for (int i = 0; i <= spreadSteps; ++i) {
      final double spread = ParameterRange.MIN_SPREAD + i * ParameterRange.STEP;
      assertTrue("spread " + spread, ParameterRange.maxMinDist(spread) <= spread);
    }
  }

  /**
   * The bound really is necessary: a spread that is not a multiple of a tenth combined with
   * a minimum distance up to the spread itself passes the argument check and then reads past
   * the end of the table row.
   */
  public void testUnboundedCombinationWouldFail() {
    try {
      Curve.curveFit(0.55f, 0.55f);
      fail("expected the lookup table to be exceeded");
    } catch (final ArrayIndexOutOfBoundsException e) {
      // exactly what maxMinDist exists to prevent
    }
  }
}
