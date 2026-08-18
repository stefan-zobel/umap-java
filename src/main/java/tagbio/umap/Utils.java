/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Utility functions.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
final class Utils {

  private Utils() {
  }

  /**
   * Get the current date and time as a string of the form
   * <code>YYYY-MM-DD hh:mm:ss</code>.
   * @return date string
   */
  static String now() {
    final StringBuilder sb = new StringBuilder();
    final Calendar cal = new GregorianCalendar();
    sb.append(cal.get(Calendar.YEAR)).append('-');
    final int month = 1 + cal.get(Calendar.MONTH);
    if (month < 10) {
      sb.append('0');
    }
    sb.append(month).append('-');
    final int date = cal.get(Calendar.DATE);
    if (date < 10) {
      sb.append('0');
    }
    sb.append(date).append(' ');
    final int hour = cal.get(Calendar.HOUR_OF_DAY);
    if (hour < 10) {
      sb.append('0');
    }
    sb.append(hour).append(':');
    final int min = cal.get(Calendar.MINUTE);
    if (min < 10) {
      sb.append('0');
    }
    sb.append(min).append(':');
    final int sec = cal.get(Calendar.SECOND);
    if (sec < 10) {
      sb.append('0');
    }
    sb.append(sec).append(' ');
    return sb.toString();
  }

  /**
   * Print a dated message on standard output.
   * @param message message to print
   */
  static void message(final String message) {
    System.out.println(now() + message);
  }

  /**
   * Wait for every one of the given tasks, and rethrow whatever any of them threw.
   * @param futures tasks to wait for
   */
  static void awaitAll(final Future<?>[] futures) {
    try {
      for (final Future<?> future : futures) {
        future.get();
      }
    } catch (final InterruptedException | ExecutionException ex) {
      throw new RuntimeException(ex);
    }
  }

  /**
   * Smallest problem handed to a thread pool, measured as <code>rows * columns</code>. The
   * work is one pass per row while creating and shutting down a pool costs a flat 0.35 ms at
   * six threads, so a floor counted in rows alone would be wrong.
   *
   * <p>Swept at six threads. On the square matrices this path always sees, break even sits
   * near 100 000: 65536 returns 0.91x and 147456 returns 1.55x. The extreme shapes are slower
   * to repay the pool, because few long rows give a pool little to divide: 32 rows of 8192
   * still returns 0.92x at 262144 and only reaches 1.63x at 524288, where every shape measured
   * is at or above break even. The floor is set there rather than at the square break even,
   * which costs at most half a millisecond on a matrix that would have won. The same element
   * count shaped as 8192 rows of 64 returns 4.02x, which is why one product rather than a row
   * count is enough.
   *
   * <p>The constant was 16384 while this method sorted whole rows. Selecting the k smallest in
   * one pass made the work some fifty times cheaper without changing what the pool costs, so
   * break even had to move by about that factor.
   * Package private so that a test can size a matrix large enough to reach the parallel path.
   */
  static final long MIN_KNN_PARALLEL_WORK = 524_288L;

  /**
   * A fast computation of knn indices.
   *
   * <p>Each row is returned in ascending distance order. Among points at an equal distance the
   * lowest index wins, so the answer is a function of the data alone and does not depend on
   * how the values happen to be ordered or on which selection algorithm is used.
   * @param instances array of shape <code>(nSamples, nFeatures)</code>
   * @param nNeighbors the number of nearest neighbors to compute for each sample in <code>instances</code>
   * @return array of shape <code>(nSamples, nNeighbors)</code> containing the indices of the <code>nNeighbours</code>
   * closest points in the dataset.
   */
  static int[][] fastKnnIndices(final Matrix instances, final int nNeighbors) {
    return fastKnnIndices(instances, nNeighbors, 1);
  }

  /**
   * As {@link #fastKnnIndices(Matrix, int)}, but spreading the rows over up to
   * <code>threads</code> workers. The result is identical to the single threaded one for any
   * thread count, ties included. A problem too small to repay a thread pool, and any value of
   * 1 or less, runs on the calling thread: no pool is created and no thread is started.
   * @param instances array of shape <code>(nSamples, nFeatures)</code>
   * @param nNeighbors the number of nearest neighbors to compute for each sample in <code>instances</code>
   * @param threads maximum number of threads to use
   * @return array of shape <code>(nSamples, nNeighbors)</code>
   */
  static int[][] fastKnnIndices(final Matrix instances, final int nNeighbors, final int threads) {
    final int rows = instances.rows();
    // Every one of these rows is replaced below, so there is nothing to allocate here.
    final int[][] knnIndices = new int[rows][];
    final int workers = (long) rows * instances.cols() < MIN_KNN_PARALLEL_WORK ? 1 : Math.min(threads, rows);
    if (workers <= 1) {
      knnRows(knnIndices, instances, nNeighbors, 0, rows);
      return knnIndices;
    }
    final ExecutorService executor = Executors.newFixedThreadPool(workers);
    try {
      final int chunkSize = (rows + workers - 1) / workers;
      final Future<?>[] futures = new Future<?>[workers];
      for (int t = 0; t < workers; ++t) {
        final int lo = t * chunkSize;
        final int hi = Math.min(lo + chunkSize, rows);
        futures[t] = executor.submit(() -> knnRows(knnIndices, instances, nNeighbors, lo, hi));
      }
      awaitAll(futures);
    } finally {
      executor.shutdown();
    }
    return knnIndices;
  }

  /**
   * Fill rows <code>[lo, hi)</code> of the knn index array.
   *
   * Contiguous ranges rather than the strides {@link PairwiseDistances} hands out: there the
   * triangular loop makes row k cost n - k, here every row costs the same one pass, so the
   * work is already flat and contiguity is worth having. Measured at n = 3000 and n = 4000
   * the two are indistinguishable, which is what a flat cost per row predicts.
   *
   * <p>The k smallest are picked up in one pass instead of sorting the row and discarding all
   * but the first k. That answered an O(n) question with an O(n log n) sort, and it did so out
   * of a copy of the row plus an identity array, 24 KB per row and 69 MB per call at
   * n = 3000. Measured at n = 3000, d = 784, k = 15, single threaded: 586 ms and 69.1 MB for
   * the full sort, 71 ms for a quickselect that keeps both of those arrays, and 10.7 ms and
   * 0.2 MB for the pass below. The allocations were the larger half of the difference, not
   * the complexity. Nothing here mutates the row, so it is read where it lies and the input
   * is untouched for that reason rather than by copying it.
   *
   * <p>Rows are disjoint and nothing is shared between workers, so the result cannot depend
   * on the worker count.
   * @param knnIndices array being filled
   * @param instances array of shape <code>(nSamples, nFeatures)</code>
   * @param nNeighbors number of nearest neighbors per sample
   * @param lo first row of the range
   * @param hi one past the last row of the range
   */
  private static void knnRows(final int[][] knnIndices, final Matrix instances, final int nNeighbors, final int lo, final int hi) {
    final int cols = instances.cols();
    final int k = Math.min(nNeighbors, cols);
    // Reused across this worker's rows, so the loop below allocates only its results. That is
    // also why size and worst are reset per row: these two outlive the row on purpose.
    final float[] bd = new float[k];
    final int[] bi = new int[k];
    for (int row = lo; row < hi; ++row) {
      final float[] r = instances.row(row);
      int size = 0;
      float worst = Float.POSITIVE_INFINITY;
      for (int j = 0; j < cols; ++j) {
        final float v = r[j];
        // Negated rather than v >= worst, so a NaN is rejected instead of shifted in.
        if (size == k && !(v < worst)) {
          continue;
        }
        // Strictly greater, so an equal distance never displaces an earlier index: among
        // equidistant points the lowest index wins, whatever order the values arrive in.
        int p = size < k ? size : k - 1;
        while (p > 0 && bd[p - 1] > v) {
          bd[p] = bd[p - 1];
          bi[p] = bi[p - 1];
          --p;
        }
        bd[p] = v;
        bi[p] = j;
        if (size < k) {
          ++size;
        }
        worst = bd[size - 1];
      }
      // Fewer columns than neighbours asked for pads with zeros, as taking a prefix of a full
      // sort did.
      knnIndices[row] = Arrays.copyOf(bi, nNeighbors);
    }
  }

  /**
   * L2 norm of a vector.
   * @param vec vector
   * @return L2 norm
   */
  static float norm(final float[] vec) {
    float result = 0;
    for (final float v : vec) {
      result += v * v;
    }
    return (float) Math.sqrt(result);
  }

  /**
   * Generate <code>nSamples</code> many integers from 0 to <code>poolSize</code> such that no
   * integer is selected twice. The duplication constraint is achieved via
   * rejection sampling.
   * @param nSamples The number of random samples to select from the pool
   * @param poolSize The size of the total pool of candidates to sample from
   * @param random Randomness source
   * @return <code>nSamples </code>randomly selected elements from the pool.
   */
  static int[] rejectionSample(final int nSamples, final int poolSize, final Random random) {
    if (nSamples > poolSize) {
      throw new IllegalArgumentException();
    }
    if (nSamples == 0) {
      return new int[0];
    }

    // O(k) sampling without replacement via a sparse index remap of a partial shuffle.
    // This avoids the previous O(k^2) duplicate checks.
    final int tableSize = tableSizeFor(nSamples << 2);
    final int[] keys = new int[tableSize];
    final int[] values = new int[tableSize];
    Arrays.fill(keys, -1);
    final int mask = tableSize - 1;
    final int[] result = new int[nSamples];
    for (int i = 0; i < nSamples; ++i) {
      final int j = i + random.nextInt(poolSize - i);
      final int vj = mapGet(keys, values, mask, j);
      final int vi = mapGet(keys, values, mask, i);
      mapPut(keys, values, mask, j, vi);
      result[i] = vj;
    }
    return result;
  }

  private static int tableSizeFor(final int required) {
    int n = 1;
    while (n < required) {
      n <<= 1;
    }
    return n;
  }

  private static int mix(final int x) {
    int h = x * 0x9E3779B9;
    h ^= h >>> 16;
    return h;
  }

  private static int mapGet(final int[] keys, final int[] values, final int mask, final int key) {
    int pos = mix(key) & mask;
    while (true) {
      final int k = keys[pos];
      if (k == -1) {
        return key;
      }
      if (k == key) {
        return values[pos];
      }
      pos = (pos + 1) & mask;
    }
  }

  private static void mapPut(final int[] keys, final int[] values, final int mask, final int key, final int value) {
    int pos = mix(key) & mask;
    while (true) {
      final int k = keys[pos];
      if (k == -1) {
        keys[pos] = key;
        values[pos] = value;
        return;
      }
      if (k == key) {
        values[pos] = value;
        return;
      }
      pos = (pos + 1) & mask;
    }
  }


// @numba.njit(parallel=true)
// def new_build_candidates(
//     current_graph,
//     n_vertices,
//     n_neighbors,
//     max_candidates,
//     rng_state,
//     rho=0.5,
// ):  # pragma: no cover
//     """Build a heap of candidate neighbors for nearest neighbor descent. For
//     each vertex the candidate neighbors are any current neighbors, and any
//     vertices that have the vertex as one of their nearest neighbors.

//     Parameters
//     ----------
//     current_graph: heap
//         The current state of the graph for nearest neighbor descent.

//     n_vertices: int
//         The total number of vertices in the graph.

//     n_neighbors: int
//         The number of neighbor edges per node in the current graph.

//     max_candidates: int
//         The maximum number of new candidate neighbors.

//     rng_state: array of int64, shape (3,)
//         The internal state of the rng

//     Returns
//     -------
//     candidate_neighbors: A heap with an array of (randomly sorted) candidate
//     neighbors for each vertex in the graph.
//     """
//     new_candidate_neighbors = make_heap(
//         n_vertices, max_candidates
//     )
//     old_candidate_neighbors = make_heap(
//         n_vertices, max_candidates
//     )

//     for i in numba.prange(n_vertices):
//         for j in range(n_neighbors):
//             if current_graph[0, i, j] < 0:
//                 continue
//             idx = current_graph[0, i, j]
//             isn = current_graph[2, i, j]
//             d = tau_rand(rng_state)
//             if tau_rand(rng_state) < rho:
//                 c = 0
//                 if isn:
//                     c += heap_push(
//                         new_candidate_neighbors,
//                         i,
//                         d,
//                         idx,
//                         isn,
//                     )
//                     c += heap_push(
//                         new_candidate_neighbors,
//                         idx,
//                         d,
//                         i,
//                         isn,
//                     )
//                 else:
//                     heap_push(
//                         old_candidate_neighbors,
//                         i,
//                         d,
//                         idx,
//                         isn,
//                     )
//                     heap_push(
//                         old_candidate_neighbors,
//                         idx,
//                         d,
//                         i,
//                         isn,
//                     )

//                 if c > 0:
//                     current_graph[2, i, j] = 0

//     return new_candidate_neighbors, old_candidate_neighbors

  /**
   * Return a submatrix given an original matrix and the indices to keep.
   *
   * <p>The companion of {@link #fastKnnIndices(Matrix, int, int)}: that returns which columns
   * are nearest, this reads what their values are. Both callers need the pair, and neither the
   * square distance matrix of a fit nor the rectangular one of a transform is treated
   * differently here, so the columns are simply whatever was measured against.
   * @param matrix Original matrix of shape <code>(nSamples, nColumns)</code>.
   * @param indicesCol Indices to keep of shape <code>(nSamples, nNeighbors)</code>.
   * Each row consists of the indices of the columns.
   * @param nNeighbors Number of neighbors.
   * @return array, shape <code>(nSamples, nNeighbors)</code>
   * The corresponding submatrix.
   */
  static float[][] submatrix(final Matrix matrix, final int[][] indicesCol, final int nNeighbors) {
    final int nSamplesTransform = matrix.rows();
    final float[][] submat = new float[nSamplesTransform][nNeighbors];
    for (int i = 0; i < nSamplesTransform; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        submat[i][j] = matrix.get(i, indicesCol[i][j]);
      }
    }
    return submat;
  }

  static Random[] splitRandom(final Random random, final int n) {
    final Random[] randoms = new Random[n];
    final long baseSeed = random.nextLong();
    for (int j = 0; j < n; ++j) {
      randoms[j] = new Random(baseSeed * (j + 1) + j); 
    }
    return randoms;
  }
}
