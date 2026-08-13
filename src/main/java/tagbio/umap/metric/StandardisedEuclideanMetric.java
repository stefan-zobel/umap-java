/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Euclidean distance standardised against a vector of standard deviations per coordinate.
 * @author Sean A. Irvine
 */
public class StandardisedEuclideanMetric extends Metric {

  private final float[] mSigma;

  public StandardisedEuclideanMetric(final float[] sigma) {
    super(false);
    mSigma = sigma;
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    //  D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;
    int i = 0;
    final int limit = x.length - 3;
    for (; i < limit; i += 4) {
      final float d0 = x[i] - y[i];
      final float d1 = x[i + 1] - y[i + 1];
      final float d2 = x[i + 2] - y[i + 2];
      final float d3 = x[i + 3] - y[i + 3];
      sum0 += d0 * d0 / mSigma[i];
      sum1 += d1 * d1 / mSigma[i + 1];
      sum2 += d2 * d2 / mSigma[i + 2];
      sum3 += d3 * d3 / mSigma[i + 3];
    }
    float result = (sum0 + sum1) + (sum2 + sum3);
    for (; i < x.length; ++i) {
      final float d = x[i] - y[i];
      result += d * d / mSigma[i];
    }
    return (float) Math.sqrt(result);
  }
}
