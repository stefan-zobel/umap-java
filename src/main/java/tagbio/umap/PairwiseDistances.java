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
   *
   * <p>The rectangular overload counts <code>xRows * yRows * columns</code>, the same unit, and
   * shares this constant. It performs every evaluation it counts where the square path performs
   * only half of them, so at an equal product it does about twice the work and repays a pool
   * sooner: the floor is conservative for that shape rather than risky. Swept at six threads
   * over square, wide and tall shapes, every one of them is at or above break even once the
   * product reaches 2e6, the closest being 8 x 4000 at d = 64 with 1.27x. The one shape that
   * would lose, a single query row against many, never reaches a pool at all because the worker
   * count is capped at the number of rows to divide.
   *
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
    return pairwiseDistances(x, y, metric, 1);
  }

  /**
   * As {@link #pairwiseDistances(Matrix, Matrix, Metric)}, but spreading the rows of
   * <code>x</code> over up to <code>threads</code> workers. The result is bit identical to the
   * single threaded one for any thread count, so this is not one of the parallel paths that
   * trade reproducibility for speed. A problem too small to repay a thread pool, and any value
   * of 1 or less, runs on the calling thread: no pool is created and no thread is started.
   * @param x matrix of instances
   * @param y matrix of instances to measure against
   * @param metric distance function
   * @param threads maximum number of threads to use
   * @return matrix of distances from every row of <code>x</code> to every row of <code>y</code>
   */
  static Matrix pairwiseDistances(final Matrix x, final Matrix y, final Metric metric, final int threads) {
    if (PrecomputedMetric.SINGLETON.equals(metric)) {
      throw new IllegalArgumentException("Cannot use this method with precomputed");
    }
    final int xn = x.rows();
    final int yn = y.rows();
    final float[][] distances = new float[xn][yn];
    final float[][] xRows = new float[xn][];
    final float[][] yRows = new float[yn][];
    // Resolving the rows once, before any worker starts, is what keeps the loop below free of
    // allocation and of a virtual call per distance.
    for (int i = 0; i < xn; ++i) {
      xRows[i] = x.row(i);
    }
    for (int j = 0; j < yn; ++j) {
      yRows[j] = y.row(j);
    }
    final int workers = (long) xn * yn * x.cols() < MIN_PARALLEL_WORK ? 1 : Math.min(threads, xn);
    if (workers <= 1) {
      // The whole range in one block is the loop this method has always run, on the caller's
      // thread.
      computeBlock(distances, xRows, yRows, metric, 0, xn);
      return new DefaultMatrix(distances);
    }
    final ExecutorService executor = Executors.newFixedThreadPool(workers);
    try {
      final Future<?>[] futures = new Future<?>[workers];
      for (int t = 0; t < workers; ++t) {
        final int lo = (int) ((long) xn * t / workers);
        final int hi = (int) ((long) xn * (t + 1) / workers);
        futures[t] = executor.submit(() -> computeBlock(distances, xRows, yRows, metric, lo, hi));
      }
      Utils.awaitAll(futures);
    } finally {
      executor.shutdown();
    }
    return new DefaultMatrix(distances);
  }

  /**
   * Fill rows <code>lo</code> up to <code>hi</code> of the rectangular distance matrix.
   *
   * Contiguous blocks rather than the strides the square case needs: every row here computes
   * the same <code>yn</code> distances, so the imbalance that striding exists to correct is
   * absent. Measured at six threads the two are indistinguishable -- 3.49x for blocks against
   * 3.42x for strides at 3000 x 3000, d = 784, 3.10x against 3.10x at 1000 x 3000, and 2.86x
   * against 2.55x at d = 64 -- so blocks were chosen for the ownership argument below rather
   * than for speed.
   *
   * A worker owns its rows outright and there is no mirror write, so every cell is written
   * exactly once by exactly one worker and no cell is ever touched by a worker that does not
   * own its row. Nothing here needs a lock or an ordering, and the result cannot depend on the
   * worker count.
   * @param distances matrix being filled
   * @param xRows rows to measure from
   * @param yRows rows to measure to
   * @param metric distance function
   * @param lo first row this worker owns
   * @param hi one past the last row this worker owns
   */
  private static void computeBlock(final float[][] distances, final float[][] xRows, final float[][] yRows, final Metric metric, final int lo, final int hi) {
    final int yn = yRows.length;
    for (int k = lo; k < hi; ++k) {
      final float[] xk = xRows[k];
      final float[] out = distances[k];
      for (int j = 0; j < yn; ++j) {
        out[j] = metric.distance(xk, yRows[j]);
      }
    }
  }

}
