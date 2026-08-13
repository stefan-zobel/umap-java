/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Manhattan distance.
 */
public final class ManhattanMetric extends Metric {

  /** Manhattan distance. */
  public static final ManhattanMetric SINGLETON = new ManhattanMetric();

  private ManhattanMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    //  D(x, y) = \sum_i |x_i - y_i|
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;
    int i = 0;
    final int limit = x.length - 3;
    for (; i < limit; i += 4) {
      sum0 += Math.abs(x[i] - y[i]);
      sum1 += Math.abs(x[i + 1] - y[i + 1]);
      sum2 += Math.abs(x[i + 2] - y[i + 2]);
      sum3 += Math.abs(x[i + 3] - y[i + 3]);
    }
    float result = (sum0 + sum1) + (sum2 + sum3);
    for (; i < x.length; ++i) {
      result += Math.abs(x[i] - y[i]);
    }
    return result;
  }
}
