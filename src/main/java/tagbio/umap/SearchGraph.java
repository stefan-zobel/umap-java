/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;

/**
 * Stores unordered pairs.
 * @author Sean A. Irvine
 * @author Richard Littin
 */
class SearchGraph {

  private static final int INITIAL_ROW_CAPACITY = 8;

  private final int[][] mRows;
  private final int[] mSizes;

  SearchGraph(final int rows) {
    mRows = new int[rows][];
    mSizes = new int[rows];
    for (int k = 0; k < rows; ++k) {
      mRows[k] = new int[INITIAL_ROW_CAPACITY];
      Arrays.fill(mRows[k], -1);
    }
  }

  private static boolean contains(final int[] row, final int size, final int value) {
    for (int i = 0; i < size; ++i) {
      if (row[i] == value) {
        return true;
      }
    }
    return false;
  }

  private void appendUnique(final int row, final int value) {
    int[] values = mRows[row];
    final int size = mSizes[row];
    if (contains(values, size, value)) {
      return;
    }
    if (size == values.length) {
      values = Arrays.copyOf(values, values.length << 1);
      Arrays.fill(values, size, values.length, -1);
      mRows[row] = values;
    }
    values[size] = value;
    mSizes[row] = size + 1;
  }

  /**
   * Set the unordered pair of instances.
   * @param x instance index
   * @param y instance index
   */
  void set(final int x, final int y) {
    appendUnique(x, y);
    appendUnique(y, x);
  }

  /**
   * Adjacency indices for an instance.
   * @param row instance number
   * @return backing adjacency array
   */
  int[] row(final int row) {
    return mRows[row];
  }

  int rowSize(final int row) {
    return mSizes[row];
  }
}
