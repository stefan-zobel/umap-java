/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Correlation distance.
 */
public final class CorrelationMetric extends Metric {

  /** Correlation distance. */
  public static final CorrelationMetric SINGLETON = new CorrelationMetric();

  private CorrelationMetric() {
    super(true);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    float muX = 0.0F;
    float muY = 0.0F;
    int i = 0;
    final int limit = x.length - 3;
    for (; i < limit; i += 4) {
      muX += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      muY += y[i] + y[i + 1] + y[i + 2] + y[i + 3];
    }
    for (; i < x.length; ++i) {
      muX += x[i];
      muY += y[i];
    }

    final float invLen = 1.0F / x.length;
    muX *= invLen;
    muY *= invLen;

    float normX = 0.0F;
    float normY = 0.0F;
    float dotProduct = 0.0F;
    i = 0;
    for (; i < limit; i += 4) {
      final float sx0 = x[i] - muX;
      final float sy0 = y[i] - muY;
      final float sx1 = x[i + 1] - muX;
      final float sy1 = y[i + 1] - muY;
      final float sx2 = x[i + 2] - muX;
      final float sy2 = y[i + 2] - muY;
      final float sx3 = x[i + 3] - muX;
      final float sy3 = y[i + 3] - muY;
      normX += sx0 * sx0 + sx1 * sx1 + sx2 * sx2 + sx3 * sx3;
      normY += sy0 * sy0 + sy1 * sy1 + sy2 * sy2 + sy3 * sy3;
      dotProduct += sx0 * sy0 + sx1 * sy1 + sx2 * sy2 + sx3 * sy3;
    }
    for (; i < x.length; ++i) {
      final float shiftedX = x[i] - muX;
      final float shiftedY = y[i] - muY;
      normX += shiftedX * shiftedX;
      normY += shiftedY * shiftedY;
      dotProduct += shiftedX * shiftedY;
    }

    if (normX == 0.0 && normY == 0.0) {
      return 0;
    } else if (dotProduct == 0.0) {
      return 1;
    } else {
      return (float) (1 - (dotProduct / Math.sqrt(normX * normY)));
    }
  }
}
