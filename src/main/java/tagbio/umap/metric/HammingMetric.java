/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Hamming distance.
 */
public final class HammingMetric extends Metric {

  /** Hamming distance. */
  public static final HammingMetric SINGLETON = new HammingMetric();

  private HammingMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    int mismatches = 0;
    int i = 0;
    final int limit = x.length - 3;
    for (; i < limit; i += 4) {
      if (x[i] != y[i]) {
        ++mismatches;
      }
      if (x[i + 1] != y[i + 1]) {
        ++mismatches;
      }
      if (x[i + 2] != y[i + 2]) {
        ++mismatches;
      }
      if (x[i + 3] != y[i + 3]) {
        ++mismatches;
      }
    }
    for (; i < x.length; ++i) {
      if (x[i] != y[i]) {
        ++mismatches;
      }
    }
    return mismatches / (float) x.length;
  }
}
