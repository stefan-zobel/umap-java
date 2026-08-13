/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Rogers Tanimoto distance.
 */
public class RogersTanimotoMetric extends Metric {

  /** Rogers Tanimoto distance. */
  public static final RogersTanimotoMetric SINGLETON = new RogersTanimotoMetric();

  RogersTanimotoMetric() {
    super(false);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    int numNotEqual = 0;
    for (int i = 0; i < x.length; ++i) {
      if ((x[i] != 0) != (y[i] != 0)) {
        ++numNotEqual;
      }
    }
    return (2 * numNotEqual) / (float) (x.length + numNotEqual);
  }
}
