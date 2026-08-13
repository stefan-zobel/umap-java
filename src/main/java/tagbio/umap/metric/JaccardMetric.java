/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Jaccard distance.
 * @author Sean A. Irvine
 */
public final class JaccardMetric extends Metric {

  /** Jaccard distance. */
  public static final JaccardMetric SINGLETON = new JaccardMetric();

  private JaccardMetric() {
    super(true);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    int numNonZero = 0;
    int numTrueTrue = 0;
    for (int i = 0; i < x.length; ++i) {
      final boolean xTrue = x[i] != 0;
      final boolean yTrue = y[i] != 0;
      if (xTrue | yTrue) {
        ++numNonZero;
        if (xTrue & yTrue) {
          ++numTrueTrue;
        }
      }
    }
    return numNonZero == 0 ? 0 : (numNonZero - numTrueTrue) / (float) numNonZero;
  }
}
