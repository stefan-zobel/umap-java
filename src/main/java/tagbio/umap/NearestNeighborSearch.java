/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;
import java.util.Random;

import tagbio.umap.metric.Metric;

/**
 * Nearest neighbor search.
 * @author Leland McInnes (Python)
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class NearestNeighborSearch {

  private final Metric mDist;

  NearestNeighborSearch(final Metric dist) {
    mDist = dist;
  }

  void treeInit(final FlatTree tree, final Matrix data, final Matrix queryPoints, final Heap heap, final Random random) {
    for (int i = 0; i < queryPoints.rows(); ++i) {
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
    for (int i = 0; i < queryPoints.rows(); ++i) {
      final int[] indices = Utils.rejectionSample(nNeighbors, data.rows(), random);
      for (final int index : indices) {
        final float d = mDist.distance(data.row(index), queryPoints.row(i));
        heap.push(i, d, index, true);
      }
    }
  }

  Heap initializedNndSearch(final Matrix data, final SearchGraph searchGraph, Heap initialization, final Matrix queryPoints) {
    final int[] visited = new int[data.rows()];
    int stamp = 1;
    for (int i = 0; i < queryPoints.rows(); ++i) {
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

    return initialization;
  }
}
