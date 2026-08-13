/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Chebyshev distance.
 */
public final class ChebyshevMetric extends Metric {

  /** Chebyshev distance. */
  public static final ChebyshevMetric SINGLETON = new ChebyshevMetric();

  private ChebyshevMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    // D(x, y) = \max_i |x_i - y_i|
    float m0 = 0.0F;
    float m1 = 0.0F;
    float m2 = 0.0F;
    float m3 = 0.0F;
    int i = 0;
    final int limit = x.length - 3;
    for (; i < limit; i += 4) {
      m0 = Math.max(m0, Math.abs(x[i] - y[i]));
      m1 = Math.max(m1, Math.abs(x[i + 1] - y[i + 1]));
      m2 = Math.max(m2, Math.abs(x[i + 2] - y[i + 2]));
      m3 = Math.max(m3, Math.abs(x[i + 3] - y[i + 3]));
    }
    float result = Math.max(Math.max(m0, m1), Math.max(m2, m3));
    for (; i < x.length; ++i) {
      result = Math.max(result, Math.abs(x[i] - y[i]));
    }
    return result;
  }
}
