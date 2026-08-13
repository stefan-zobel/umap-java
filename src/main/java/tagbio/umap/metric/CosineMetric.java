/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap.metric;

/**
 * Cosine distance.
 * @author Sean A. Irvine
 */
public final class CosineMetric extends Metric {

  /** Cosine distance. */
  public static final CosineMetric SINGLETON = new CosineMetric();

  private CosineMetric() {
    super(true);
  }

  @Override
  public float distance(final float[] x, final float[] y) {
    double dot0 = 0.0;
    double dot1 = 0.0;
    double dot2 = 0.0;
    double dot3 = 0.0;
    double nx0 = 0.0;
    double nx1 = 0.0;
    double nx2 = 0.0;
    double nx3 = 0.0;
    double ny0 = 0.0;
    double ny1 = 0.0;
    double ny2 = 0.0;
    double ny3 = 0.0;
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
      dot0 += x0 * y0;
      dot1 += x1 * y1;
      dot2 += x2 * y2;
      dot3 += x3 * y3;
      nx0 += x0 * x0;
      nx1 += x1 * x1;
      nx2 += x2 * x2;
      nx3 += x3 * x3;
      ny0 += y0 * y0;
      ny1 += y1 * y1;
      ny2 += y2 * y2;
      ny3 += y3 * y3;
    }
    double result = (dot0 + dot1) + (dot2 + dot3);
    double normX = (nx0 + nx1) + (nx2 + nx3);
    double normY = (ny0 + ny1) + (ny2 + ny3);
    for (; i < x.length; ++i) {
      final float xv = x[i];
      final float yv = y[i];
      result += xv * yv;
      normX += xv * xv;
      normY += yv * yv;
    }
    if (normX == 0.0 && normY == 0.0) {
      return 0;
    } else if (normX == 0.0 || normY == 0.0) {
      return 1;
    } else {
      return (float) (1 - (result / Math.sqrt(normX * normY)));
    }
  }
}
