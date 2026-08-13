/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import tagbio.umap.metric.Metric;
import tagbio.umap.metric.PrecomputedMetric;

/**
 * Compute pairwise distances between instances using a specified metric.
 * @author Sean A. Irvine
 * @author Richard Littin
 */
public class PairwiseDistances {

  // replacement for sklearn.pairwise_distances

  private PairwiseDistances() { }

  static Matrix pairwiseDistances(final Matrix x, final Metric metric) {
    if (PrecomputedMetric.SINGLETON.equals(metric)) {
      return x;
    }
    final int n = x.rows();
    final float[][] distances = new float[n][n];
    final float[][] rows = new float[n][];
    for (int i = 0; i < n; ++i) {
      rows[i] = x.row(i);
    }
    for (int k = 0; k < n; ++k) {
      final float[] xk = rows[k];
      for (int j = k; j < n; ++j) {
        final float d = metric.distance(xk, rows[j]);
        distances[k][j] = d;
        distances[j][k] = d;
      }
    }
    return new DefaultMatrix(distances);
  }

  static Matrix pairwiseDistances(final Matrix x, final Matrix y, final Metric metric) {
    if (PrecomputedMetric.SINGLETON.equals(metric)) {
      throw new IllegalArgumentException("Cannot use this method with precomputed");
    }
    final int xn = x.rows();
    final int yn = y.rows();
    final float[][] distances = new float[xn][yn];
    final float[][] xRows = new float[xn][];
    final float[][] yRows = new float[yn][];
    for (int i = 0; i < xn; ++i) {
      xRows[i] = x.row(i);
    }
    for (int j = 0; j < yn; ++j) {
      yRows[j] = y.row(j);
    }
    for (int k = 0; k < xn; ++k) {
      final float[] xk = xRows[k];
      for (int j = 0; j < yn; ++j) {
        distances[k][j] = metric.distance(xk, yRows[j]);
      }
    }
    return new DefaultMatrix(distances);
  }

}
