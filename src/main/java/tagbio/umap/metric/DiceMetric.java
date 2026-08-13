/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Dice distance.
 * @author Sean A. Irvine
 */
public final class DiceMetric extends Metric {

  /** Dice distance. */
  public static final DiceMetric SINGLETON = new DiceMetric();

  private DiceMetric() {
    super(true);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    int numTrueTrue = 0;
    int numNotEqual = 0;
    for (int i = 0; i < x.length; ++i) {
      final boolean xTrue = x[i] != 0;
      final boolean yTrue = y[i] != 0;
      if (xTrue == yTrue) {
        if (xTrue) {
          ++numTrueTrue;
        }
      } else {
        ++numNotEqual;
      }
    }
    return numNotEqual == 0 ? 0 : numNotEqual / (float) (2 * numTrueTrue + numNotEqual);
  }
}
