/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import tagbio.umap.metric.Metric;

/**
 * Nearest neighbor search.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class NearestNeighborSearch {

  /**
   * Smallest problem handed to a thread pool by {@link #initializedNndSearch}, measured as
   * <code>queryRows * heapWidth * columns</code>. The work is one graph walk per query row while
   * creating and shutting down a pool costs a flat 0.35 ms at six threads, and a walk is far more
   * expensive than the row loops the other floors guard: at d = 784 and a heap width of 60 as few
   * as three query rows already repay a pool.
   *
   * <p>A floor counted in rows alone would be wrong by nearly two orders of magnitude, because
   * per-row work scales with both the dimension and the heap width. Break even swept at six
   * threads over a 6000 row model: 3, 13 and 28 query rows at d = 784, 64 and 8 with a heap width
   * of 60, and 8, 60 and 200 with a width of 20. As a product those span 13 000 to 141 000, which
   * is why the product is what is counted.
   *
   * <p>No single product can be exact here: the cost of one work unit varies sevenfold with the
   * dimension, 4.3 ns at d = 784 against 30 ns at d = 8, because the heap push and the stamp check
   * per candidate are fixed cost. 65536 splits the difference, and both directions of the error
   * are small: two query rows at d = 784 are allowed at 0.84x, wasting about 0.08 ms, while
   * d = 8 at a width of 60 waits until 137 rows where break even was 28, forgoing about 1.2 ms.
   * Both are within what a pool costs, and the low dimensions are where a walk is cheapest in
   * absolute terms -- 14 ms per 1000 rows at d = 8 -- so they are the corner least worth chasing.
   *
   * <p>{@link NearestNeighborDescent#initialiseSearch} shares this constant. It is a different
   * loop, but it is one pass per query row over a heap of the same width against a model of the
   * same dimension, and its sweep lands in the same place: break even at 4 to 32 query rows at a
   * width of 60 and 8 to 128 at a width of 20, a product span of 15 000 to 376 000 against this
   * loop's 13 000 to 141 000. At 65536 the worst case there is 1.9 ms forgone at d = 784 with a
   * narrow heap, and at most 0.1 ms wasted anywhere -- the same trade this constant already makes
   * for the walk. A second constant within noise of this one would only invite the two to drift
   * apart.
   *
   * Package private so that a test can size a query matrix large enough to reach the parallel
   * path at all.
   */
  static final long MIN_SEARCH_PARALLEL_WORK = 65_536L;

  private final Metric mDist;

  NearestNeighborSearch(final Metric dist) {
    mDist = dist;
  }

  void treeInit(final FlatTree tree, final Matrix data, final Matrix queryPoints, final Heap heap, final Random random) {
    treeInit(tree, data, queryPoints, heap, random, 0, queryPoints.rows());
  }

  /**
   * Seed the heap rows <code>[lo, hi)</code> from the leaf each query row descends to.
   *
   * <p>Rows are independent: each writes only into its own heap row, {@link FlatTree} is immutable
   * and the tree descent only reads it. The <code>random</code> is the one thing a caller must not
   * share between threads -- {@link FlatTree#searchFlatTree} draws from it whenever a point sits
   * on a hyperplane -- which is why {@link NearestNeighborDescent#initialiseSearch} hands each
   * worker its own and why that method is not exact above one thread.
   */
  void treeInit(final FlatTree tree, final Matrix data, final Matrix queryPoints, final Heap heap, final Random random, final int lo, final int hi) {
    for (int i = lo; i < hi; ++i) {
      final int[] indices = tree.searchFlatTree(queryPoints.row(i), random);
      for (final int index : indices) {
        if (index < 0) {
          continue;
        }
        final float d = mDist.distance(data.row(index), queryPoints.row(i));
        heap.push(i, d, index, true);
      }
    }
  }

  void randomInit(final int nNeighbors, final Matrix data, final Matrix queryPoints, final Heap heap, final Random random) {
    randomInit(nNeighbors, data, queryPoints, heap, random, 0, queryPoints.rows());
  }

  /**
   * Seed the heap rows <code>[lo, hi)</code> with <code>nNeighbors</code> randomly chosen
   * instances each. As {@link #treeInit(FlatTree, Matrix, Matrix, Heap, Random, int, int)}: rows
   * are independent and only the <code>random</code> must not be shared.
   */
  void randomInit(final int nNeighbors, final Matrix data, final Matrix queryPoints, final Heap heap, final Random random, final int lo, final int hi) {
    for (int i = lo; i < hi; ++i) {
      final int[] indices = Utils.rejectionSample(nNeighbors, data.rows(), random);
      for (final int index : indices) {
        final float d = mDist.distance(data.row(index), queryPoints.row(i));
        heap.push(i, d, index, true);
      }
    }
  }

  /**
   * Walk the search graph from an initialized heap, refining the neighbours of every query row.
   *
   * <p>The rows are divided among at most <code>threads</code> workers. The result is identical
   * to the single threaded one for any thread count, bit for bit: see {@link #searchBlock}. A
   * problem too small to repay a thread pool, and any value of 1 or less, runs on the calling
   * thread; no pool is created and no thread is started.
   * @param data the instances the model was fit on
   * @param searchGraph adjacency of the fitted neighbour graph
   * @param initialization heap of starting candidates, one row per query point, refined in place
   * @param queryPoints the instances being transformed
   * @param threads maximum number of threads to use
   * @return the refined heap, the same object that was passed in
   */
  Heap initializedNndSearch(final Matrix data, final SearchGraph searchGraph, final Heap initialization, final Matrix queryPoints, final int threads) {
    final int rows = queryPoints.rows();
    // An empty query matrix has no heap row to read the width from, and nothing to divide.
    final int workers = rows == 0 || (long) rows * initialization.indices()[0].length * data.cols() < MIN_SEARCH_PARALLEL_WORK
      ? 1 : Math.min(threads, rows);
    if (workers <= 1) {
      // The whole range in one block is the loop this method has always run, on the caller's
      // thread.
      searchBlock(data, searchGraph, initialization, queryPoints, 0, rows);
      return initialization;
    }
    final ExecutorService executor = Executors.newFixedThreadPool(workers);
    try {
      final Future<?>[] futures = new Future<?>[workers];
      for (int t = 0; t < workers; ++t) {
        final int lo = (int) ((long) rows * t / workers);
        final int hi = (int) ((long) rows * (t + 1) / workers);
        futures[t] = executor.submit(() -> searchBlock(data, searchGraph, initialization, queryPoints, lo, hi));
      }
      Utils.awaitAll(futures);
    } finally {
      executor.shutdown();
    }
    return initialization;
  }

  /**
   * Walk the graph for query rows <code>[lo, hi)</code>.
   *
   * <p>A worker owns its heap rows outright: {@link Heap#smallestFlagged} and
   * {@link Heap#uncheckedHeapPush} touch only row <code>i</code>, the search graph and the data
   * are read only and complete before any worker starts, and the one thing the serial loop shares
   * -- the visited stamp array -- is allocated here, one per worker. Nothing in the walk draws
   * from a {@link Random}, so nothing depends on how the rows were divided and the candidates of
   * a row are still visited in the same order. The result is therefore bit identical to the
   * serial one at any thread count, which a test asserts at five of them rather than assuming.
   *
   * <p>Contiguous blocks rather than strides. Walk length varies per query row, so imbalance was
   * a real possibility here and was measured rather than argued away: at six threads and d = 784
   * blocks reached 2.80x against 2.56x for strides at 64 query rows, 3.67x against 3.35x at 256
   * and 3.30x against 2.94x at 1000, and at d = 64 blocks won at every size. The absolute ratios
   * move with the machine's state -- a later session measured the same code at about two thirds
   * of these -- but the ordering held in both. Blocks also cross one row boundary per worker
   * instead of every row, which is where false sharing between neighbouring heap rows would come
   * from.
   *
   * <p>The stamp array costs <code>4 * data.rows()</code> bytes per worker, 24 KB on a 6000 row
   * model, and the serial path already allocates one of them per call. A model of a million rows
   * at sixteen workers would allocate 64 MB per call, but its walks are correspondingly more
   * expensive, so the ratio does not deteriorate.
   */
  private void searchBlock(final Matrix data, final SearchGraph searchGraph, final Heap initialization, final Matrix queryPoints, final int lo, final int hi) {
    final int[] visited = new int[data.rows()];
    int stamp = 1;
    for (int i = lo; i < hi; ++i) {
      if (stamp == Integer.MAX_VALUE) {
        Arrays.fill(visited, 0);
        stamp = 1;
      }

      for (final int t : initialization.indices()[i]) {
        if (t >= 0) {
          visited[t] = stamp;
        }
      }

      final float[] queryRow = queryPoints.row(i);

      while (true) {

        // Find smallest flagged vertex
        final int vertex = initialization.smallestFlagged(i);

        if (vertex == -1) {
          break;
        }
        final int[] neighbors = searchGraph.row(vertex);
        final int degree = searchGraph.rowSize(vertex);
        for (int n = 0; n < degree; ++n) {
          final int candidate = neighbors[n];
          if (candidate == vertex || candidate < 0 || visited[candidate] == stamp) {
            continue;
          }
          final float d = mDist.distance(data.row(candidate), queryRow);
          initialization.uncheckedHeapPush(i, d, candidate, true);
          visited[candidate] = stamp;
        }
      }
      ++stamp;
    }
  }
}
