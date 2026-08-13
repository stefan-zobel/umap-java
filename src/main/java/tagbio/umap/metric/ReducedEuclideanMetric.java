/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Reduced Euclidean distance.
 * @author Leland McInnes
 * @author Sean A. Irvine
 * @author Richard Littin
 */
public final class ReducedEuclideanMetric extends Metric {

  /** Reduced Euclidean distance. */
  public static final ReducedEuclideanMetric SINGLETON = new ReducedEuclideanMetric();

  private ReducedEuclideanMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    //  D(x, y) = \sum_i (x_i - y_i)^2
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
      sum0 += d0 * d0;
      sum1 += d1 * d1;
      sum2 += d2 * d2;
      sum3 += d3 * d3;
    }
    float result = (sum0 + sum1) + (sum2 + sum3);
    for (; i < x.length; ++i) {
      final float d = x[i] - y[i];
      result += d * d;
    }
    return result;
  }
}
