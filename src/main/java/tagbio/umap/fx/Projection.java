package tagbio.umap.fx;

/**
 * A computed embedding together with the data it was computed from and the parameters that
 * produced it, so that neither the point labels nor the reported settings can drift away
 * from the picture on screen.
 */
final class Projection {

  private final PointData mData;
  private final float[][] mEmbedding;
  private final int mNeighbours;
  private final float mMinDist;
  private final float mSpread;
  private final int mThreads;

  Projection(final PointData data, final float[][] embedding, final int neighbours, final float minDist,
             final float spread, final int threads) {
    mData = data;
    mEmbedding = embedding;
    mNeighbours = neighbours;
    mMinDist = minDist;
    mSpread = spread;
    mThreads = threads;
  }

  /**
   * The source data, including sample and class names.
   * @return the projected data
   */
  PointData getData() {
    return mData;
  }

  /**
   * The embedding, one row per sample with at least two columns.
   * @return the embedded coordinates
   */
  float[][] getEmbedding() {
    return mEmbedding;
  }

  /**
   * The neighbour count this embedding was computed with.
   * @return number of nearest neighbours
   */
  int getNeighbours() {
    return mNeighbours;
  }

  /**
   * The minimum distance this embedding was computed with.
   * @return the minimum distance
   */
  float getMinDist() {
    return mMinDist;
  }

  /**
   * The spread this embedding was computed with.
   * @return the spread
   */
  float getSpread() {
    return mSpread;
  }

  /**
   * The thread count this embedding was computed with, which decides whether it could be
   * reproduced at all: only a single threaded run repeats.
   * @return the number of threads used
   */
  int getThreads() {
    return mThreads;
  }
}
