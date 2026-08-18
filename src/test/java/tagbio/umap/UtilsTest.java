package tagbio.umap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;

import junit.framework.TestCase;

public class UtilsTest extends TestCase {

  public void testRejectionSample() {
    final Random r = new Random();
    final int[] rs = Utils.rejectionSample(20, 100, r);
    assertEquals(20, rs.length);
    final TreeSet<Integer> uniq = new TreeSet<>();
    for (final int v : rs) {
      assertTrue(v >= 0 && v < 100);
      uniq.add(v);
    }
    assertEquals(20, uniq.size());
    try {
      Utils.rejectionSample(5, 2, r);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testSplitRandom() {
    Random[] randoms = Utils.splitRandom(new Random(543), 5);
    assertEquals(5, randoms.length);
    final int[] expected = new int[]{-1797116241, -80536573, -1257863196, 1902860816, 160052042, -993477666, -1141936413, 1152672626, -749860475, -1591028618};
    for (int i = 0; i < randoms.length; i++) {
      assertEquals(expected[i], randoms[i].nextInt());
    }
    randoms = Utils.splitRandom(new Random(543), 7);
    assertEquals(7, randoms.length);
    for (int i = randoms.length - 1; i >= 0; i--) {
      assertEquals(expected[i], randoms[i].nextInt());
    }
    randoms = Utils.splitRandom(new Random(543), 10);
    assertEquals(10, randoms.length);
    for (int i = 0; i < randoms.length; i++) {
      assertEquals(expected[i], randoms[i].nextInt());
    }
        randoms = Utils.splitRandom(new Random(345), 10);
    assertEquals(10, randoms.length);
    for (int i = 0; i < randoms.length; i++) {
      assertNotSame(expected[i], randoms[i].nextInt());
    }
  }

  public void testNow() {
    assertTrue(Utils.now().matches("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9] "));
  }

  public void testFastKnnIndices() {
    final Matrix m = new DefaultMatrix(new float[][] {{1, 2}, {0, 1}, {1, 0}, {0, 0}, {2, 1}});
    final int[][] knn = Utils.fastKnnIndices(m, 2);
    assertEquals("[[0, 1], [0, 1], [1, 0], [0, 1], [1, 0]]", Arrays.deepToString(knn));
  }

  /**
   * Columns needed for a matrix of <code>rows</code> rows to clear
   * {@link Utils#MIN_KNN_PARALLEL_WORK}. Derived rather than written down, so that raising the
   * floor cannot quietly drop these tests onto the serial path where they would compare it
   * with itself.
   */
  private static int knnFixtureCols(final int rows) {
    return (int) (Utils.MIN_KNN_PARALLEL_WORK / rows) + 1;
  }

  /**
   * A matrix of distinct values, wide enough to reach the parallel path. A shuffled
   * permutation of the whole numbers rather than random draws: every value is distinct by
   * construction and no rejection loop is needed, which matters because the floor is large.
   */
  private static Matrix knnFixture(final int rows) {
    return knnFixture(rows, knnFixtureCols(rows));
  }

  /** As {@link #knnFixture(int)}, for a shape chosen by the caller. */
  private static Matrix knnFixture(final int rows, final int cols) {
    final int count = rows * cols;
    final int[] order = MathUtils.identity(count);
    final Random random = new Random(7);
    for (int i = count - 1; i > 0; --i) {
      final int j = random.nextInt(i + 1);
      final int t = order[i];
      order[i] = order[j];
      order[j] = t;
    }
    final float[][] data = new float[rows][cols];
    for (int i = 0, k = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        // Distinct throughout, so the k smallest of a row are an unambiguous set and no tie
        // could hide a worker that produced them in the wrong order.
        data[i][j] = order[k++];
      }
    }
    return new DefaultMatrix(data);
  }

  /**
   * Splitting the rows over threads must reproduce the single threaded result exactly: rows
   * are disjoint and a worker's selection buffers are its own, so there is nothing here that a
   * worker count is entitled to change.
   */
  public void testFastKnnIndicesParallelMatchesSerial() {
    final int rows = 97;   // divisible by none of the worker counts below
    final Matrix m = knnFixture(rows);
    final int k = 5;
    final int[][] serial = Utils.fastKnnIndices(m, k);
    for (final int threads : new int[] {2, 3, 4, 7, 16}) {
      final int[][] parallel = Utils.fastKnnIndices(m, k, threads);
      assertEquals("threads=" + threads, Arrays.deepToString(serial), Arrays.deepToString(parallel));
    }
    // Two implementations that agreed on garbage would still pass the loop above.
    for (int row = 0; row < rows; ++row) {
      assertEquals(k, serial[row].length);
      for (int j = 1; j < k; ++j) {
        assertTrue("row " + row + " is not in ascending distance order",
          m.get(row, serial[row][j - 1]) < m.get(row, serial[row][j]));
      }
    }
  }

  /** Records who read each row, so that a claim about threads can be checked directly. */
  private static final class ThreadRecordingMatrix extends DefaultMatrix {
    private final List<Thread> mCallers = new ArrayList<>();

    private ThreadRecordingMatrix(final float[][] data) {
      super(data);
    }

    @Override
    float[] row(final int row) {
      synchronized (mCallers) {
        mCallers.add(Thread.currentThread());
      }
      return super.row(row);
    }
  }

  /**
   * The single threaded path must not merely produce the same numbers, it must not hand the
   * work to anyone else: callers rely on threads = 1 meaning no pool is created and no thread
   * is started. The same holds for a matrix too small to repay a pool, whatever is asked for.
   */
  public void testFastKnnIndicesSingleThreadRunsOnTheCallingThread() {
    final int rows = 64;
    final ThreadRecordingMatrix big = new ThreadRecordingMatrix(new float[rows][knnFixtureCols(rows)]);
    Utils.fastKnnIndices(big, 3, 1);
    assertFalse("no row was read, the test proves nothing", big.mCallers.isEmpty());
    for (final Thread caller : big.mCallers) {
      assertSame("work was handed to another thread", Thread.currentThread(), caller);
    }

    // Sized from the floor rather than written down, so it stays below it however it moves.
    final ThreadRecordingMatrix small = new ThreadRecordingMatrix(new float[rows][knnFixtureCols(rows) / 2]);
    Utils.fastKnnIndices(small, 3, 6);
    assertFalse("no row was read, the test proves nothing", small.mCallers.isEmpty());
    for (final Thread caller : small.mCallers) {
      assertSame("a pool was started for a matrix too small to repay it", Thread.currentThread(), caller);
    }
  }

  /**
   * Among points at an equal distance the lowest index wins. That is a guarantee of the
   * method, not an accident of how the values happen to be laid out, so it is pinned here:
   * before the k smallest were selected in one pass an unstable quicksort decided this and the
   * answer depended on the sort.
   */
  public void testFastKnnIndicesBreaksTiesByLowestIndex() {
    final Matrix m = new DefaultMatrix(new float[][] {
      {5, 1, 1, 1, 5, 1},   // the three smallest are the ones at 1, 2 and 3
      {0, 0, 0, 9, 9, 9},
      {3, 3, 3, 3, 3, 3},   // every column is tied, so the first three win
    });
    assertEquals("[[1, 2, 3], [0, 1, 2], [0, 1, 2]]", Arrays.deepToString(Utils.fastKnnIndices(m, 3)));

    // The same on the parallel path, over a matrix where every value repeats every eighth
    // column: the k smallest are the lowest indices carrying the smallest value.
    final int rows = 97;
    final int cols = knnFixtureCols(rows);
    final float[][] data = new float[rows][cols];
    for (final float[] row : data) {
      for (int j = 0; j < cols; ++j) {
        row[j] = j % 8;
      }
    }
    final Matrix tied = new DefaultMatrix(data);
    for (final int threads : new int[] {1, 2, 3, 4, 7, 16}) {
      final int[][] knn = Utils.fastKnnIndices(tied, 5, threads);
      for (int row = 0; row < rows; ++row) {
        assertEquals("threads=" + threads + " row " + row,
          "[0, 8, 16, 24, 32]", Arrays.toString(knn[row]));
      }
    }
  }

  /**
   * Whichever equidistant point is chosen, the k distances carried back have to be the k
   * smallest of the row. This is the property that makes the tie rule harmless downstream, so
   * it is asserted directly rather than inferred from the embedding tests.
   */
  public void testFastKnnIndicesReturnsTheKSmallestDistances() {
    final int rows = 97;
    final int cols = knnFixtureCols(rows);
    final int k = 15;
    final Random random = new Random(11);
    final float[][] data = new float[rows][cols];
    for (final float[] row : data) {
      for (int j = 0; j < cols; ++j) {
        // A small range over many columns, so ties at the k-th distance are the rule here
        // rather than the exception.
        row[j] = random.nextInt(20);
      }
    }
    final Matrix m = new DefaultMatrix(data);
    for (final int threads : new int[] {1, 4, 7}) {
      final int[][] knn = Utils.fastKnnIndices(m, k, threads);
      for (int row = 0; row < rows; ++row) {
        final float[] sorted = Arrays.copyOf(data[row], cols);
        Arrays.sort(sorted);
        for (int j = 0; j < k; ++j) {
          assertEquals("threads=" + threads + " row " + row + " neighbour " + j,
            sorted[j], m.get(row, knn[row][j]), 0.0F);
        }
      }
    }
  }

  /**
   * Every other fixture here is square, but {@code Umap.transform} hands this method a
   * rectangular matrix: rows are the new instances and columns the training ones. A confusion
   * of rows with columns, in the worker split or in the row loop, is invisible on a square
   * matrix, so both a wide and a tall shape are checked -- and the tall one is the case where
   * many workers each hold short rows.
   */
  public void testFastKnnIndicesOnARectangularMatrix() {
    final int k = 5;
    final int wide = knnFixtureCols(41);
    final int[][] shapes = {{41, wide}, {wide, 41}};
    for (final int[] shape : shapes) {
      final int rows = shape[0];
      final int cols = shape[1];
      final Matrix m = knnFixture(rows, cols);
      final int[][] serial = Utils.fastKnnIndices(m, k);
      for (final int threads : new int[] {1, 2, 3, 7, 16}) {
        final int[][] parallel = Utils.fastKnnIndices(m, k, threads);
        assertEquals(rows + "x" + cols + " threads=" + threads,
          Arrays.deepToString(serial), Arrays.deepToString(parallel));
      }
      // Two implementations that agreed on garbage would still pass the loop above.
      for (int row = 0; row < rows; ++row) {
        assertEquals(k, serial[row].length);
        final float[] sorted = Arrays.copyOf(m.row(row), cols);
        Arrays.sort(sorted);
        for (int j = 0; j < k; ++j) {
          assertEquals(rows + "x" + cols + " row " + row + " neighbour " + j,
            sorted[j], m.get(row, serial[row][j]), 0.0F);
        }
      }
    }

    // Ties in every row, so the lowest index has to win in both shapes as well.
    for (final int[] shape : shapes) {
      final float[][] data = new float[shape[0]][shape[1]];
      for (final float[] row : data) {
        for (int j = 0; j < row.length; ++j) {
          row[j] = j % 8;
        }
      }
      for (final int[] row : Utils.fastKnnIndices(new DefaultMatrix(data), k, 7)) {
        assertEquals("[0, 8, 16, 24, 32]", Arrays.toString(row));
      }
    }
  }

  /**
   * Asking for more neighbours than there are columns pads with zeros, which is what taking a
   * prefix of a full sort used to do. Nothing else covers it.
   */
  public void testFastKnnIndicesWithMoreNeighboursThanColumns() {
    final Matrix m = new DefaultMatrix(new float[][] {{7, 4}, {1, 6}});
    assertEquals("[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]", Arrays.deepToString(Utils.fastKnnIndices(m, 5)));
  }

  public void testL2Norm() {
    final float[] vec = new float[2];
    assertEquals(0.0, Utils.norm(vec), 1e-6);
    vec[0] = 1;
    assertEquals(1.0, Utils.norm(vec), 1e-6);
    vec[1] = 1;
    assertEquals(Math.sqrt(2), Utils.norm(vec), 1e-6);
    vec[0] = 3;
    vec[1] = 4;
    assertEquals(5.0, Utils.norm(vec), 1e-6);
  }
}

