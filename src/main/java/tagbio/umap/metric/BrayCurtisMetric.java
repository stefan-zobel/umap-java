/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Bray Curtis distance.
 */
public final class BrayCurtisMetric extends Metric {

  /** Bray Curtis distance. */
  public static final BrayCurtisMetric SINGLETON = new BrayCurtisMetric();

  private BrayCurtisMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    float n0 = 0.0F;
    float n1 = 0.0F;
    float n2 = 0.0F;
    float n3 = 0.0F;
    float d0 = 0.0F;
    float d1 = 0.0F;
    float d2 = 0.0F;
    float d3 = 0.0F;
    int i = 0;
    final int limit = x.length - 3;
    for (; i < limit; i += 4) {
      final float x0 = x[i];
      final float y0 = y[i];
      final float x1 = x[i + 1];
      final float y1 = y[i + 1];
      final float x2 = x[i + 2];
      final float y2 = y[i + 2];
      final float x3 = x[i + 3];
      final float y3 = y[i + 3];
      n0 += Math.abs(x0 - y0);
      n1 += Math.abs(x1 - y1);
      n2 += Math.abs(x2 - y2);
      n3 += Math.abs(x3 - y3);
      d0 += Math.abs(x0 + y0);
      d1 += Math.abs(x1 + y1);
      d2 += Math.abs(x2 + y2);
      d3 += Math.abs(x3 + y3);
    }
    float numerator = (n0 + n1) + (n2 + n3);
    float denominator = (d0 + d1) + (d2 + d3);
    for (; i < x.length; ++i) {
      numerator += Math.abs(x[i] - y[i]);
      denominator += Math.abs(x[i] + y[i]);
    }
    return denominator > 0 ? numerator / denominator : 0;
  }
}
