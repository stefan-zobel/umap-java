/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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

  /**
   * Smallest problem handed to a thread pool, measured as <code>rows * rows * columns</code>.
   * The work is O(n<sup>2</sup>/2 &middot; d) while creating and shutting down a pool costs a
   * flat 0.35 ms at six threads, so a floor counted in rows alone would be wrong: break even
   * sits at n = 48 for d = 784, at n = 145 for d = 64 and at n = 384 for d = 4. The largest of
   * those three is 1.8e6, so 2e6 is at or above break even for all of them. Deliberately on
   * the conservative side; below the threshold the whole phase is under a millisecond.
   * Package private so that a test can size a matrix large enough to reach the parallel path
   * at all.
   */
  static final long MIN_PARALLEL_WORK = 2_000_000L;

  static Matrix pairwiseDistances(final Matrix x, final Metric metric) {
    return pairwiseDistances(x, metric, 1);
  }

  /**
   * As {@link #pairwiseDistances(Matrix, Metric)}, but spreading the rows over up to
   * <code>threads</code> workers. The result is bit identical to the single threaded one for
   * any thread count, so this is not one of the parallel paths that trade reproducibility for
   * speed. A problem too small to repay a thread pool, and any value of 1 or less, runs on the
   * calling thread: no pool is created and no thread is started.
   * @param x matrix of instances
   * @param metric distance function
   * @param threads maximum number of threads to use
   * @return matrix of pairwise distances
   */
  static Matrix pairwiseDistances(final Matrix x, final Metric metric, final int threads) {
    if (PrecomputedMetric.SINGLETON.equals(metric)) {
      return x;
    }
    final int n = x.rows();
    final float[][] distances = new float[n][n];
    final float[][] rows = new float[n][];
    for (int i = 0; i < n; ++i) {
      rows[i] = x.row(i);
    }
    final int workers = (long) n * n * x.cols() < MIN_PARALLEL_WORK ? 1 : Math.min(threads, n);
    if (workers <= 1) {
      // Stride 1 from row 0 is the loop this method has always run, on the caller's thread.
      computeRows(distances, rows, metric, 0, 1);
      return new DefaultMatrix(distances);
    }
    final ExecutorService executor = Executors.newFixedThreadPool(workers);
    try {
      final Future<?>[] futures = new Future<?>[workers];
      for (int t = 0; t < workers; ++t) {
        final int start = t;
        futures[t] = executor.submit(() -> computeRows(distances, rows, metric, start, workers));
      }
      Utils.awaitAll(futures);
    } finally {
      executor.shutdown();
    }
    return new DefaultMatrix(distances);
  }

  /**
   * Fill rows <code>start, start + stride, ...</code> of the distance matrix.
   *
   * Handing out rows in strides rather than in contiguous blocks is what keeps the workers
   * evenly fed: row k computes n - k distances, so a block of low rows is a multiple of the
   * work of a block of high ones. Measured at n = 3000, d = 784 on six threads, contiguous
   * blocks reach 1.86x and strides 2.25x, and at two threads the gap is 1.31x against 1.96x.
   *
   * Every cell is written exactly once, and by exactly one worker, so nothing here needs a
   * lock or an ordering and the result cannot depend on the worker count. The mirror write
   * <code>distances[j][k]</code> is the only write into a row another worker may own, and the
   * owner of row j never writes that cell itself because its own loop starts at j.
   * @param distances matrix being filled
   * @param metric distance function
   * @param start first row this worker owns
   * @param stride distance between the rows this worker owns
   */
  private static void computeRows(final float[][] distances, final float[][] rows, final Metric metric, final int start, final int stride) {
    final int n = rows.length;
    for (int k = start; k < n; k += stride) {
      final float[] xk = rows[k];
      for (int j = k; j < n; ++j) {
        final float d = metric.distance(xk, rows[j]);
        distances[k][j] = d;
        distances[j][k] = d;
      }
    }
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
