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
   * A matrix of distinct values, wide enough that <code>rows * columns</code> clears
   * {@link Utils#MIN_KNN_PARALLEL_WORK}. Without that, every call below would fall to the
   * serial path and the comparison would be against itself.
   */
  private static Matrix knnFixture(final int rows) {
    final int cols = (int) (Utils.MIN_KNN_PARALLEL_WORK / rows) + 1;
    final Random random = new Random(7);
    final float[][] data = new float[rows][cols];
    final TreeSet<Float> seen = new TreeSet<>();
    for (final float[] row : data) {
      for (int j = 0; j < cols; ++j) {
        // Distinct throughout, so the k smallest of a row are an unambiguous set and no tie
        // could hide a worker that produced them in the wrong order.
        float v;
        do {
          v = random.nextFloat();
        } while (!seen.add(v));
        row[j] = v;
      }
    }
    return new DefaultMatrix(data);
  }

  /**
   * Splitting the rows over threads must reproduce the single threaded result exactly: rows
   * are disjoint, each is copied before it is sorted, and Sort keeps no state, so there is
   * nothing here that a worker count is entitled to change.
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
    final ThreadRecordingMatrix big = new ThreadRecordingMatrix(new float[128][128]);
    Utils.fastKnnIndices(big, 3, 1);
    assertFalse("no row was read, the test proves nothing", big.mCallers.isEmpty());
    for (final Thread caller : big.mCallers) {
      assertSame("work was handed to another thread", Thread.currentThread(), caller);
    }

    final ThreadRecordingMatrix small = new ThreadRecordingMatrix(new float[16][16]);
    Utils.fastKnnIndices(small, 3, 6);
    assertFalse("no row was read, the test proves nothing", small.mCallers.isEmpty());
    for (final Thread caller : small.mCallers) {
      assertSame("a pool was started for a matrix too small to repay it", Thread.currentThread(), caller);
    }
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

