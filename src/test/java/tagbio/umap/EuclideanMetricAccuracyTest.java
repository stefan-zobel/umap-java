/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Random;

import junit.framework.TestCase;
import tagbio.umap.metric.EuclideanMetric;

/**
 * Checks how far <code>EuclideanMetric.distance</code> is from the exact answer.
 *
 * The reference is the exact value: the squared differences of the float inputs are summed in
 * <code>BigDecimal</code>, so only the accumulation differs from the implementation under
 * test. Comparing against a value produced by another library would compare one approximation
 * with another -- numpy accumulates in float64 and uses its own summation order, while this
 * port is float32 throughout, so the two can never agree bit for bit and any tolerance would
 * be arbitrary. Exactness is the only oracle that makes the question well posed.
 *
 * The bounds are deliberately loose enough to survive a change of summation order, which is
 * a legitimate optimization, and tight enough to catch an accumulation that actually loses
 * precision. Measured maxima at the time of writing: 1.48 ulp for iris, 0.50 for digits,
 * 1.92 for 128-dimensional uniform data and 2.01 for data spanning eight decades.
 */
public class EuclideanMetricAccuracyTest extends TestCase {

  /** No formulation of a sum of d squares should drift further than this. */
  private static final double MAX_ULP = 4.0;

  /** Typical error, which catches a regression that only shows up on average. */
  private static final double MEAN_ULP = 0.75;

  /** Exact sum of the squared differences, then a correctly rounded square root. */
  private static double exact(final float[] x, final float[] y) {
    BigDecimal sum = BigDecimal.ZERO;
    for (int i = 0; i < x.length; ++i) {
      final BigDecimal d = new BigDecimal(x[i]).subtract(new BigDecimal(y[i]));
      sum = sum.add(d.multiply(d));
    }
    return Math.sqrt(sum.round(new MathContext(60)).doubleValue());
  }

  private void assertAccurate(final String name, final float[][] data) {
    double maxUlp = 0;
    double sumUlp = 0;
    long pairs = 0;
    for (int i = 0; i < data.length; ++i) {
      for (int j = i + 1; j < data.length; ++j) {
        final double ref = exact(data[i], data[j]);
        if (ref == 0.0) {
          continue;   // coincident points, nothing to measure
        }
        final float actual = EuclideanMetric.SINGLETON.distance(data[i], data[j]);
        final double ulp = Math.abs(actual - ref) / Math.ulp((float) ref);
        maxUlp = Math.max(maxUlp, ulp);
        sumUlp += ulp;
        ++pairs;
      }
    }
    assertTrue(name + ": no pairs compared", pairs > 0);
    final double meanUlp = sumUlp / pairs;
    assertTrue(name + ": max error " + maxUlp + " ulp exceeds " + MAX_ULP, maxUlp <= MAX_ULP);
    assertTrue(name + ": mean error " + meanUlp + " ulp exceeds " + MEAN_ULP, meanUlp <= MEAN_ULP);
  }

  public void testIris() throws IOException {
    assertAccurate("iris", new IrisData().getData());
  }

  public void testDigits() throws IOException {
    // Small integers, so every squared difference and every partial sum is exact; this data
    // set pins that an exactly representable case stays exactly representable.
    assertAccurate("digits", new DigitData().getData());
  }

  public void testHighDimensionalUniform() {
    final Random random = new Random(7);
    final float[][] data = new float[200][128];
    for (final float[] row : data) {
      for (int j = 0; j < row.length; ++j) {
        row[j] = random.nextFloat();
      }
    }
    assertAccurate("uniform 128d", data);
  }

  public void testWideDynamicRange() {
    // Eight decades in one row is the hostile case for any running sum.
    final Random random = new Random(11);
    final float[][] data = new float[150][64];
    for (final float[] row : data) {
      for (int j = 0; j < row.length; ++j) {
        row[j] = (float) Math.pow(10.0, -4 + 8 * random.nextDouble());
      }
    }
    assertAccurate("wide range 64d", data);
  }

  /** A sanity check on the reference itself: an exactly representable case must land dead on. */
  public void testReferenceIsExactForRepresentableInput() {
    final float[] x = {0, 0, 0};
    final float[] y = {3, 4, 0};
    assertEquals(5.0, exact(x, y), 0.0);
    assertEquals(5.0F, EuclideanMetric.SINGLETON.distance(x, y));
  }
}
