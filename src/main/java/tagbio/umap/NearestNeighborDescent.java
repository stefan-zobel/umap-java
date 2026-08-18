/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import tagbio.umap.metric.Metric;

/**
 * Nearest neighbor descent for a specified distance metric.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class NearestNeighborDescent {

  final Metric mMetric;
  boolean mVerbose;

  /**
   * Construct a nearest neighbor descent object for the given metric.
   * @param metric distance function
   */
  NearestNeighborDescent(final Metric metric) {
    mMetric = metric;
  }

  void setVerbose(boolean flag) {
    mVerbose = flag;
  }

  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rpTreeInit, final int nIters, final List<FlatTree> forest) {
    return descent(data, nNeighbors, random, maxCandidates, rpTreeInit, nIters, forest, 0.001F, 0.5F);
  }

  Heap descent(final Matrix data, final int nNeighbors, final Random random, final int maxCandidates, final boolean rpTreeInit, final int nIters, final List<FlatTree> forest, final float delta, final float rho) {
    final int nVertices = data.rows();
    final float[][] rows = new float[nVertices][];
    for (int i = 0; i < nVertices; ++i) {
      rows[i] = data.row(i);
    }
    final Heap currentGraph = new Heap(data.rows(), nNeighbors);
    for (int i = 0; i < nVertices; ++i) {
      final float[] iRow = rows[i];
      for (final int index : Utils.rejectionSample(nNeighbors, nVertices, random)) {
        final float d = mMetric.distance(iRow, rows[index]);
        currentGraph.push(i, d, index, true);
        currentGraph.push(index, d, i, true);
      }
    }
    UmapProgress.update();

    if (rpTreeInit) {
      for (final FlatTree tree : forest) {
        for (final int[] leaf : tree.getIndices()) {
          for (int i = 0; i < leaf.length; ++i) {
            final float[] iRow = rows[leaf[i]];
            for (int j = i + 1; j < leaf.length; ++j) {
              final float d = mMetric.distance(iRow, rows[leaf[j]]);
              currentGraph.push(leaf[i], d, leaf[j], true);
              currentGraph.push(leaf[j], d, leaf[i], true);
            }
          }
        }
      }
    }
    UmapProgress.update();

    final boolean[] rejectStatus = new boolean[maxCandidates];
    for (int n = 0; n < nIters; ++n) {
      if (mVerbose) {
        Utils.message("NearestNeighborDescent: " + (n + 1) + " / " + nIters);
      }

      final Heap candidateNeighbors = currentGraph.buildCandidates(nVertices, nNeighbors, maxCandidates, random);

      int c = 0;
      for (int i = 0; i < nVertices; ++i) {
        for (int j = 0; j < maxCandidates; ++j) {
          rejectStatus[j] = random.nextFloat() < rho;
        }

        for (int j = 0; j < maxCandidates; ++j) {
          final int p = candidateNeighbors.index(i, j);
          if (p < 0) {
            continue;
          }
          for (int k = 0; k <= j; ++k) {
            final int q = candidateNeighbors.index(i, k);
            if (q < 0 || (rejectStatus[j] && rejectStatus[k]) || (!candidateNeighbors.isNew(i, j) && !candidateNeighbors.isNew(i, k))) {
              continue;
            }

            final float d = mMetric.distance(rows[p], rows[q]);
            if (currentGraph.push(p, d, q, true)) {
              ++c;
            }
            if (currentGraph.push(q, d, p, true)) {
              ++c;
            }
          }
        }
      }

      if (c <= delta * nNeighbors * nVertices) {
        UmapProgress.update(nIters - n);
        break;
      }
      UmapProgress.update();
    }
    return currentGraph.deheapSort();
  }


  /**
   * Build the heap of starting candidates a search walks from: random instances for every query
   * row, then the leaf each query row descends to in every tree of the forest.
   *
   * <p>The rows are divided among at most <code>threads</code> workers, each with its own random
   * stream. <b>Above one thread the result therefore depends on the thread count</b>, as the
   * parallel descent's already does: the stream decides which side of a hyperplane a point on it
   * falls, and so which leaf is reached. At one thread, and for a problem too small to repay a
   * pool, this is exactly the loop it has always been, on the calling thread and drawing from the
   * caller's random. Measured, a different stream moves 5 to 6 percent of the neighbours a
   * transform finds and leaves their total distance unchanged to five decimal places.
   * @param forest random projection forest, may be null
   * @param data the instances the model was fit on
   * @param queryPoints the instances being transformed
   * @param nNeighbors width of the heap to build
   * @param nn search whose metric is used
   * @param random source of randomness
   * @param threads maximum number of threads to use
   * @return the heap of starting candidates
   */
  static Heap initialiseSearch(final List<FlatTree> forest, final Matrix data, final Matrix queryPoints, final int nNeighbors, final NearestNeighborSearch nn, final Random random, final int threads) {
    final int rows = queryPoints.rows();
    final Heap results = new Heap(rows, nNeighbors);
    final int workers = (long) rows * nNeighbors * data.cols() < NearestNeighborSearch.MIN_SEARCH_PARALLEL_WORK
      ? 1 : Math.min(threads, rows);
    if (workers <= 1) {
      // The whole range in one block is the loop this method has always run, on the caller's
      // thread and out of the caller's random.
      initialiseBlock(forest, data, queryPoints, nNeighbors, nn, random, results, 0, rows);
      return results;
    }
    final Random[] randoms = Utils.splitRandom(random, workers);
    final ExecutorService executor = Executors.newFixedThreadPool(workers);
    try {
      final Future<?>[] futures = new Future<?>[workers];
      for (int t = 0; t < workers; ++t) {
        final int lo = (int) ((long) rows * t / workers);
        final int hi = (int) ((long) rows * (t + 1) / workers);
        final Random own = randoms[t];
        futures[t] = executor.submit(() -> initialiseBlock(forest, data, queryPoints, nNeighbors, nn, own, results, lo, hi));
      }
      Utils.awaitAll(futures);
    } finally {
      executor.shutdown();
    }
    return results;
  }

  /**
   * Seed the heap rows <code>[lo, hi)</code>.
   *
   * <p>The random initialization runs over the whole range before the first tree does, and one
   * tree over the whole range before the next: a heap row therefore receives its pushes in the
   * same order it would serially, so nothing but the random stream distinguishes a worker's rows
   * from the ones the serial path produces. A worker owns its rows outright, and {@link Heap#push}
   * locks the row it writes to in any case.
   */
  private static void initialiseBlock(final List<FlatTree> forest, final Matrix data, final Matrix queryPoints, final int nNeighbors, final NearestNeighborSearch nn, final Random random, final Heap results, final int lo, final int hi) {
    nn.randomInit(nNeighbors, data, queryPoints, results, random, lo, hi);
    if (forest != null) {
      for (final FlatTree tree : forest) {
        nn.treeInit(tree, data, queryPoints, results, random, lo, hi);
      }
    }
  }
}
