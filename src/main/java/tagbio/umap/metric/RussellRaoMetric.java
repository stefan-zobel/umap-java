/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Russel Rao distance.
 */
public final class RussellRaoMetric extends Metric {

  /** Russel Rao distance. */
  public static final RussellRaoMetric SINGLETON = new RussellRaoMetric();

  private RussellRaoMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    int numTrueTrue = 0;
    boolean allTrueTrue = true;
    for (int i = 0; i < x.length; ++i) {
      final boolean xTrue = x[i] != 0;
      final boolean yTrue = y[i] != 0;
      if (xTrue && yTrue) {
        ++numTrueTrue;
      } else {
        allTrueTrue = false;
      }
    }
    return allTrueTrue ? 0 : (x.length - numTrueTrue) / (float) x.length;
  }
}
