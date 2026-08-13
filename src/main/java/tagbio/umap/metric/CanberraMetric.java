/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Canberra distance.
 */
public final class CanberraMetric extends Metric {

  /** Canberra distance. */
  public static final CanberraMetric SINGLETON = new CanberraMetric();

  private CanberraMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;
    int i = 0;
    final int limit = x.length - 3;
    for (; i < limit; i += 4) {
      final float x0 = x[i];
      final float y0 = y[i];
      final float d0 = Math.abs(x0) + Math.abs(y0);
      if (d0 > 0) {
        sum0 += Math.abs(x0 - y0) / d0;
      }
      final float x1 = x[i + 1];
      final float y1 = y[i + 1];
      final float d1 = Math.abs(x1) + Math.abs(y1);
      if (d1 > 0) {
        sum1 += Math.abs(x1 - y1) / d1;
      }
      final float x2 = x[i + 2];
      final float y2 = y[i + 2];
      final float d2 = Math.abs(x2) + Math.abs(y2);
      if (d2 > 0) {
        sum2 += Math.abs(x2 - y2) / d2;
      }
      final float x3 = x[i + 3];
      final float y3 = y[i + 3];
      final float d3 = Math.abs(x3) + Math.abs(y3);
      if (d3 > 0) {
        sum3 += Math.abs(x3 - y3) / d3;
      }
    }
    float result = (sum0 + sum1) + (sum2 + sum3);
    for (; i < x.length; ++i) {
      final float denominator = Math.abs(x[i]) + Math.abs(y[i]);
      if (denominator > 0) {
        result += Math.abs(x[i] - y[i]) / denominator;
      }
    }
    return result;
  }
}
