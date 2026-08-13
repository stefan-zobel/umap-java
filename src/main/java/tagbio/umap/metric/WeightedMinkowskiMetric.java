/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Weighted Minkowski distance.
 */
public class WeightedMinkowskiMetric extends Metric {

  private static final double EPS = 1e-12;

  private final double mPower;
  private final double mInvPower;
  private final float[] mWeights;
  private final boolean mIsL1;
  private final boolean mIsL2;

  public WeightedMinkowskiMetric(final double power, final float[] weights) {
    super(false);
    mPower = power;
    mInvPower = 1.0 / power;
    mWeights = weights;
    mIsL1 = Math.abs(power - 1.0) < EPS;
    mIsL2 = Math.abs(power - 2.0) < EPS;
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    // D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}
    if (mIsL1) {
      float sum0 = 0.0F;
      float sum1 = 0.0F;
      float sum2 = 0.0F;
      float sum3 = 0.0F;
      int i = 0;
      final int limit = x.length - 3;
      for (; i < limit; i += 4) {
        sum0 += mWeights[i] * Math.abs(x[i] - y[i]);
        sum1 += mWeights[i + 1] * Math.abs(x[i + 1] - y[i + 1]);
        sum2 += mWeights[i + 2] * Math.abs(x[i + 2] - y[i + 2]);
        sum3 += mWeights[i + 3] * Math.abs(x[i + 3] - y[i + 3]);
      }
      float result = (sum0 + sum1) + (sum2 + sum3);
      for (; i < x.length; ++i) {
        result += mWeights[i] * Math.abs(x[i] - y[i]);
      }
      return result;
    }

    if (mIsL2) {
      float sum0 = 0.0F;
      float sum1 = 0.0F;
      float sum2 = 0.0F;
      float sum3 = 0.0F;
      int i = 0;
      final int limit = x.length - 3;
      for (; i < limit; i += 4) {
        final float d0 = mWeights[i] * (x[i] - y[i]);
        final float d1 = mWeights[i + 1] * (x[i + 1] - y[i + 1]);
        final float d2 = mWeights[i + 2] * (x[i + 2] - y[i + 2]);
        final float d3 = mWeights[i + 3] * (x[i + 3] - y[i + 3]);
        sum0 += d0 * d0;
        sum1 += d1 * d1;
        sum2 += d2 * d2;
        sum3 += d3 * d3;
      }
      float result = (sum0 + sum1) + (sum2 + sum3);
      for (; i < x.length; ++i) {
        final float d = mWeights[i] * (x[i] - y[i]);
        result += d * d;
      }
      return (float) Math.sqrt(result);
    }

    double result = 0.0;
    for (int i = 0; i < x.length; ++i) {
      result += Math.pow(mWeights[i] * Math.abs(x[i] - y[i]), mPower);
    }
    return (float) Math.pow(result, mInvPower);
  }
}
