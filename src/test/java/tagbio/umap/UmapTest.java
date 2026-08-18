/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import junit.framework.TestCase;
import tagbio.umap.metric.EuclideanMetric;
import tagbio.umap.metric.MinkowskiMetric;
import tagbio.umap.metric.PrecomputedMetric;

/**
 * Tests the corresponding class.
 */
public class UmapTest extends TestCase {

  private static final int AGREEMENT_K = 15;

  /**
   * Fraction of each point's <code>k</code> nearest neighbors in the embedding that carry the
   * same label as the point itself. Chance level is one over the number of classes.
   *
   * This replaces the frozen sums over the embedding that used to stand here. Such a sum is a
   * chaotic function of the last bits of every distance the SGD ever computed: a one-ulp
   * change anywhere moves the optimizer into a different basin and the sum jumps, even though
   * the embedding is just as good. It broke twice for exactly that reason and never once
   * caught a defect. Agreement measures what the embedding is actually for -- keeping similar
   * points together -- and is stable against numerical drift while still collapsing towards
   * chance level if the algorithm genuinely breaks.
   */
  private static double neighbourLabelAgreement(final float[][] embedding, final int[] labels, final int k) {
    final float[] best = new float[k];
    final int[] bestIdx = new int[k];
    int hits = 0;
    int total = 0;
    for (int i = 0; i < embedding.length; ++i) {
      Arrays.fill(best, Float.MAX_VALUE);
      Arrays.fill(bestIdx, -1);
      for (int j = 0; j < embedding.length; ++j) {
        if (j == i) {
          continue;
        }
        float d = 0;
        for (int c = 0; c < embedding[i].length; ++c) {
          final float diff = embedding[i][c] - embedding[j][c];
          d += diff * diff;
        }
        if (d < best[k - 1]) {
          int p = k - 1;
          while (p > 0 && best[p - 1] > d) {
            best[p] = best[p - 1];
            bestIdx[p] = bestIdx[p - 1];
            --p;
          }
          best[p] = d;
          bestIdx[p] = j;
        }
      }
      for (final int idx : bestIdx) {
        if (idx >= 0) {
          if (labels[idx] == labels[i]) {
            ++hits;
          }
          ++total;
        }
      }
    }
    return (double) hits / total;
  }

  private void assertAgreementAtLeast(final double expected, final float[][] embedding, final int[] labels) {
    final double actual = neighbourLabelAgreement(embedding, labels, AGREEMENT_K);
    assertTrue("neighbour label agreement " + actual + " below " + expected, actual >= expected);
  }

  private static float[][] toFloat(final double[][] x) {
    final float[][] res = new float[x.length][x[0].length];
    for (int i = 0; i < x.length; ++i) {
      for (int j = 0; j < x[i].length; ++j) {
        res[i][j] = (float) x[i][j];
      }
    }
    return res;
  }

  // The thresholds below sit under the worst of seeds {42, 1, 7, 123, 98765, -4444, 10101}
  // with several points of margin, and far above the chance level of the data set.

  public void testIris() throws IOException {
    final Data data = new IrisData();
    final Umap umap = new Umap();
    umap.setVerbose(true);
    final float[][] d = data.getData();
    final long start = System.currentTimeMillis();
    final float[][] matrix = umap.fitTransform(d);
    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 1000.0) + " s");
    assertEquals(150, matrix.length);
    assertEquals(2, matrix[0].length);
    // Three classes, so chance is 0.333; observed across seeds 0.938 to 0.951.
    assertAgreementAtLeast(0.90, matrix, data.getSampleClassIndex());

    final float[][] t = umap.transform(d);
    assertEquals(150, t.length);
    assertEquals(2, t[0].length);
    // Re-embedding the training data is a little looser: observed 0.923 to 0.947.
    assertAgreementAtLeast(0.88, t, data.getSampleClassIndex());
  }

  public void testIrisViaDouble() throws IOException {
    final Data data = new IrisData();
    final Umap umap = new Umap();
    umap.setVerbose(true);
    final float[][] d = data.getData();
    final double[][] dd = new double[d.length][d[0].length];
    for (int k = 0; k < d.length; ++k) {
      for (int j = 0; j < d[0].length; ++j) {
        dd[k][j] = d[k][j];
      }
    }
    final double[][] matrix = umap.fitTransform(dd);
    assertEquals(150, matrix.length);
    assertEquals(2, matrix[0].length);
    assertAgreementAtLeast(0.90, toFloat(matrix), data.getSampleClassIndex());
  }

  public void testDigits() throws IOException {
    final Data data = new DigitData();
    final Umap umap = new Umap();
    umap.setVerbose(true);
    umap.setNumberComponents(3);
    final float[][] d = data.getData();
    final long start = System.currentTimeMillis();
    final float[][] matrix = umap.fitTransform(d);
    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 1000.0) + " s");
    assertEquals(1797, matrix.length);
    assertEquals(3, matrix[0].length);
    // Ten classes, so chance is 0.1; observed across seeds 0.974 to 0.981.
    assertAgreementAtLeast(0.95, matrix, data.getSampleClassIndex());
  }

//  public void testMammoth() throws IOException {
//    final Data data = new MammothData();
//    final float[][] d = data.getData();
//    final long start = System.currentTimeMillis();
//    final Umap umap = new Umap();
//    umap.setVerbose(true);
//    umap.setNumberComponents(2);
//    umap.setNumberNearestNeighbours(100);
//    final float[][] matrix = umap.fitTransform(d);
//    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 1000.0) + " s");
//    assertEquals(10000, matrix.length);
//    assertEquals(2, matrix[0].length);
//    final int[] classIndexes = data.getSampleClassIndex();
//    for (int r = 0; r < matrix.length; ++r) {
//      System.out.println(matrix[r][0] + " " + matrix[r][1] + " " + classIndexes[r]);
//    }
//  }

//  public void testGenes() throws IOException {
//    final Data data = new GeneData();
//    final Umap umap = new Umap();
//    umap.setVerbose(true);
//    umap.setNumberComponents(2);
//    final float[][] d = data.getData();
//    final long start = System.currentTimeMillis();
//    final float[][] matrix = umap.fitTransform(d);
//    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 1000.0) + " s");
//    assertEquals(5902, matrix.length);
//    assertEquals(2, matrix[0].length);
//    assertEquals(-5602.466796875, MathUtils.sum(matrix), 1e-4);
////    final int[] classIndexes = data.getSampleClassIndex();
////    for (int r = 0; r < matrix.length; ++r) {
////      System.out.println(matrix[r][0] + " " + matrix[r][1] + " " + classIndexes[r]);
////    }
//  }

  private int[] primes(final int m) {
    final List<Integer> primes = new ArrayList<>();
    final boolean[] state = new boolean[m];
    Arrays.fill(state, true);
    state[0] = false;
    state[1] = false;
    for (int k = 2; k < m; ++k) {
      if (state[k]) {
        primes.add(k);
      }
      for (int j = k; j < m; j += k) {
        state[j] = false;
      }
    }
    final int[] res = new int[primes.size()];
    for (int k = 0; k < res.length; ++k) {
      res[k] = primes.get(k);
    }
    return res;
  }

  private float[][] factorizations(final int[] omega, final int m) {
    final int[] primes = primes(m);
    final float[][] data = new float[omega.length][primes.length + 1];
    for (int k = 0; k < omega.length; ++k) {
      int s = k;
      for (int j = 0; j < primes.length && s > 1; ++j) {
        final int p = primes[j];
        while (s % p == 0) {
          ++data[k][j];
          ++omega[k];
          s /= p;
        }
      }
      data[k][primes.length] = s;
    }
    return data;
  }

  public void testPrimes() {
    //final int[] omega = new int[1000000];
    //final float[][] d = factorizations(omega, 1000);
    final int[] omega = new int[1000];
    final float[][] d = factorizations(omega, 100);
    final long start = System.currentTimeMillis();
    final Umap umap = new Umap();
    umap.setVerbose(true);
    umap.setNumberComponents(2);
    //umap.setThreads(4);
    final float[][] matrix = umap.fitTransform(d);
    System.out.println("UMAP time: " + Math.round((System.currentTimeMillis() - start) / 1000.0) + " s");
    assertEquals(1000, matrix.length);
    assertEquals(2, matrix[0].length);
    // Labelled by the number of prime factors, which is real but much weaker structure than
    // iris or digits: chance is 0.1, observed across seeds 0.520 to 0.546.
    assertAgreementAtLeast(0.45, matrix, omega);
  }

  public void testFindABParams() throws IOException {
    final Data data = new IrisData();
    final Umap umap = new Umap();

    for (float spread : new float[]{-1.234F, 0.0F, 2.0F, 0.49F, 1.51F}) {
      umap.setSpread(spread);
      try {
        umap.fitTransform(data.getData());
        fail("Accepted bad spread " + spread);
      } catch (IllegalArgumentException iae) {
        assertTrue(iae.getMessage().contains("spread"));
      }
    }
    umap.setSpread(1.0F);
    for (float dist : new float[]{-1.234F, -0.01F, 1.01F, 1.51F}) {
      try {
        umap.setMinDist(dist);
        umap.fitTransform(data.getData());
        fail("Accepted bad dist " + dist);
      } catch (IllegalArgumentException iae) {
        assertTrue(iae.getMessage(), iae.getMessage().contains("ist"));
      }
    }
  }

  public void testNegativeOp() {
    final Umap umap = new Umap();
    try {
      umap.setSetOpMixRatio(-1.0F);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testTooLargeOp() {
    final Umap umap = new Umap();
    try {
      umap.setSetOpMixRatio(1.5F);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testNegativeMinDist() {
    final Umap umap = new Umap();
    try {
      umap.setMinDist(-1);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testNegativeComponents() {
    final Umap umap = new Umap();
    try {
      umap.setNumberComponents(-1);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testTooSmallNeighbours() {
    final Umap umap = new Umap();
    try {
      umap.setNumberNearestNeighbours(0);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testBadMetric() {
    final Umap umap = new Umap();
    try {
      umap.setMetric("no-such-metric");
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testNegativeLearningRate() {
    final Umap umap = new Umap();
    try {
      umap.setLearningRate(-1.5F);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testNegativeRepulsion() {
    final Umap umap = new Umap();
    try {
      umap.setRepulsionStrength(-0.5F);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testBadLocalConnectivity() {
    final Umap umap = new Umap();
    for (final float bad : new float[] {-0.5F, -1.0F, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
      try {
        umap.setLocalConnectivity(bad);
        fail("Accepted local connectivity " + bad);
      } catch (final IllegalArgumentException e) {
        assertTrue(e.getMessage(), e.getMessage().contains("Local connectivity"));
      }
    }
    // The int overload delegates, so it rejects negatives too.
    try {
      umap.setLocalConnectivity(-1);
      fail("Accepted local connectivity -1");
    } catch (final IllegalArgumentException e) {
      // expected
    }
    // Zero and fractional values are legitimate.
    umap.setLocalConnectivity(0.0F);
    umap.setLocalConnectivity(1.5F);
    umap.setLocalConnectivity(2);
  }

  public void testNegativeSampleRate() {
    final Umap umap = new Umap();
    try {
      umap.setNegativeSampleRate(-1);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  public void testNegativeEpochs() {
    final Umap umap = new Umap();
    try {
      umap.setNumberEpochs(-2);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }

  private void assertArrayEquals(final double[] expected, final float[] actual) {
    if (expected.length != actual.length) {
      fail("Lengths mismatch: expected=" + expected.length + " actual=" + actual.length);
    }
    for (int k = 0; k < expected.length; ++k) {
      assertEquals("Mismatch at index " + k + " expected=" + expected[k] + " actual=" + actual[k], expected[k], actual[k], 1e-6);
    }
  }

  private void assertArrayEquals(final double[][] expected, final float[][] actual) {
    if (expected.length != actual.length) {
      fail("Lengths mismatch: expected=" + expected.length + " actual=" + actual.length);
    }
    for (int k = 0; k < expected.length; ++k) {
      assertArrayEquals(expected[k], actual[k]);
    }
  }

  public void testSmoothKnnDist() throws IOException {
    final Matrix distances = new IrisData(true).getDistances();
    //System.out.println(distances.toStringNumpy());
    final float[][] smooth21 = Umap.smoothKnnDist(distances.toArray(), 2, 1);
    // Comparison values from Python
    assertArrayEquals(new double[] {0.00326393, 0.00322133, 0.00330938, 0.0026791, 0.00247916, 0.00266279, 0.00299635, 0.00269514, 0.00280051, 0.00712167}, smooth21[0]);
    assertArrayEquals(new double[] {0.5385164, 0.5385164, 0.509902, 4.003748, 3.6166282, 4.1641326, 4.853864, 4.1904655, 4.4170127, 6.3450766}, smooth21[1]);
    final float[][] smooth42 = Umap.smoothKnnDist(distances.toArray(), 4, 2);
    assertArrayEquals(new double[] {0.71514893, 0.25, 0.25, 0.0026791, 0.00247916, 0.00266279, 0.00299635, 0.00269514, 0.00280051, 0.00712167}, smooth42[0]);
    assertArrayEquals(new double[] {0.509902, 0.30000022, 0.30000022, 4.0963397, 3.6864617, 4.236744, 4.9020405, 4.134005, 4.402272, 5.916925}, smooth42[1]);
  }

  /** Rho for a single row, which is the part of smoothKnnDist that localConnectivity drives. */
  private float rho(final float[][] distances, final float localConnectivity) {
    return Umap.smoothKnnDist(distances, 4.0F, localConnectivity)[1][0];
  }

  /**
   * A fractional local connectivity interpolates between neighbors. This is only expressible
   * since the parameter was widened from int to float; with an int the interpolation term is
   * always zero and both branches below collapse.
   */
  public void testSmoothKnnDistFractionalLocalConnectivity() {
    final float[][] d = {{0, 2, 4, 8}};   // the leading zero is the self match

    // Whole numbers select a neighbor outright.
    assertEquals(0.0F, rho(d, 0.0F), 1e-6);
    assertEquals(2.0F, rho(d, 1.0F), 1e-6);
    assertEquals(4.0F, rho(d, 2.0F), 1e-6);

    // Below one, the distance to the nearest neighbor is scaled down.
    assertEquals(1.0F, rho(d, 0.5F), 1e-6);

    // Above one, interpolate linearly towards the next neighbor: 2 + 0.5 * (4 - 2).
    assertEquals(3.0F, rho(d, 1.5F), 1e-6);
    assertEquals(6.0F, rho(d, 2.5F), 1e-6);   // 4 + 0.5 * (8 - 4)
  }

  /**
   * The positive distances are addressed by their ordinal among the positive entries, not by
   * an offset from the first one, so a row whose zeros are not a leading run still works.
   * Guards the regression that a contiguous-suffix assumption introduced here once before.
   */
  public void testSmoothKnnDistUnsortedRow() {
    final float[][] d = {{4, 0, 2, 8}};   // positives in row order: 4, 2, 8

    assertEquals(4.0F, rho(d, 1.0F), 1e-6);
    assertEquals(2.0F, rho(d, 2.0F), 1e-6);
    assertEquals(3.0F, rho(d, 1.5F), 1e-6);   // 4 + 0.5 * (2 - 4)

    // Fewer positives than required falls back to the largest of them.
    assertEquals(8.0F, rho(d, 4.0F), 1e-6);
  }

  /**
   * When nNeighbors reaches or exceeds the number of rows it is truncated to rows - 1, so
   * every such setting has to produce the same embedding. If any step downstream used the
   * requested count instead of the truncated one, these two runs would diverge.
   *
   * Note this exercises the small-data branch. The large-data branch cannot be covered the
   * same way: it needs at least SMALL_PROBLEM_THRESHOLD rows and an equally large neighbour
   * count, which means a 4096 by 4095 heap and a nearest neighbor descent over it.
   */
  public void testNeighbourCountExceedingDataSizeIsTruncated() throws IOException {
    final Data data = new IrisData();
    final float[][] d = data.getData();
    final int rows = d.length;

    final Umap exactly = new Umap();
    exactly.setSeed(42);
    exactly.setNumberNearestNeighbours(rows);
    final float[][] a = exactly.fitTransform(d);

    final Umap beyond = new Umap();
    beyond.setSeed(42);
    beyond.setNumberNearestNeighbours(2 * rows);
    final float[][] b = beyond.fitTransform(d);

    assertEquals(rows, a.length);
    assertEquals(2, a[0].length);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < a[i].length; ++j) {
        assertEquals("row " + i + " column " + j, a[i][j], b[i][j]);
      }
    }
    // Still a usable embedding, not just a reproducible one.
    assertAgreementAtLeast(0.88, a, data.getSampleClassIndex());
  }

  /**
   * The neighbour count handed to fuzzySimplicialSet is not cosmetic: it becomes the target
   * of the smooth knn distance search, so passing the untruncated count while the distance
   * arrays hold the truncated one changes the graph. This is what makes the previous test's
   * invariant meaningful.
   */
  public void testFuzzySimplicialSetDependsOnNeighbourCount() throws IOException {
    final Matrix distances = new IrisData(true).getDistances();
    final IndexedDistances nn = Umap.nearestNeighbors(distances, 3, PrecomputedMetric.SINGLETON, false, null, 1, false);

    final float[][] matching = Umap.fuzzySimplicialSet(distances, 3, null, PrecomputedMetric.SINGLETON,
      nn.getIndices(), nn.getDistances(), false, 1, 1, 1, false).toArray();
    final float[][] mismatched = Umap.fuzzySimplicialSet(distances, 2, null, PrecomputedMetric.SINGLETON,
      nn.getIndices(), nn.getDistances(), false, 1, 1, 1, false).toArray();

    boolean differs = false;
    for (int i = 0; i < matching.length && !differs; ++i) {
      for (int j = 0; j < matching[i].length; ++j) {
        if (matching[i][j] != mismatched[i][j]) {
          differs = true;
          break;
        }
      }
    }
    assertTrue("neighbour count does not reach the graph", differs);
  }

  public void testNearestNeighborsPrecomputed() throws IOException {
    final Matrix distances = new IrisData(true).getDistances();
    final IndexedDistances id = Umap.nearestNeighbors(distances, 2, PrecomputedMetric.SINGLETON, false, null, 1, false);
    // Comparison values from Python
    assertTrue(Arrays.deepEquals(new int[][] {{0, 2}, {1, 2}, {2, 1}, {3, 5}, {4, 3}, {5, 3}, {6, 5}, {7, 8}, {8, 7}, {9, 2}}, id.getIndices()));
    assertArrayEquals(new double[][] {{0, 0.509902}, {0, 0.30000022}, {0, 0.30000022}, {0, 0.26457536}, {0, 0.64031225}, {0, 0.26457536}, {0, 0.86023235}, {0, 0.51961535}, {0, 0.51961535}, {0, 5.8360944}}, id.getDistances());
    assertTrue(id.getForest().isEmpty());
  }

  public void testComputeMembershipStrengths() throws IOException {
    final Matrix distances = new IrisData(true).getDistances();
    final float[][] sigmaRhos = Umap.smoothKnnDist(distances.toArray(), 2, 1);
    final IndexedDistances id = Umap.nearestNeighbors(distances, 2, PrecomputedMetric.SINGLETON, false, null, 1, false);
    final CooMatrix m = Umap.computeMembershipStrengths(id.getIndices(), id.getDistances(), sigmaRhos[0], sigmaRhos[1], distances.rows(), distances.cols());
    // Comparison values from Python
    // The next three lines are order dependent in the CooMatrix, so not ideal for comparison
//    assertTrue(Arrays.equals(new int[]{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9}, m.mRow));
//    assertTrue(Arrays.equals(new int[]{0, 2, 1, 2, 1, 2, 3, 5, 3, 4, 3, 5, 5, 6, 7, 8, 7, 8, 2, 9}, m.mCol));
//    assertArrayEquals(new double[]{0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0}, m.mData);
    assertArrayEquals(new double[][]{
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    }, m.toArray());
  }

  public void testFuzzySimplicialSet() throws IOException {
    final Matrix distances = new IrisData(true).getDistances();
    final Matrix m = Umap.fuzzySimplicialSet(distances, 2, null, PrecomputedMetric.SINGLETON, null, null, false, 1, 1, 1, false);
    // Comparison values from Python
    /*
    [[0.         0.         0.99999222 0.         0.         0.          0.         0.         0.         0.        ]
     [0.         0.         1.         0.         0.         0.          0.         0.         0.         0.        ]
     [0.99999222 1.         0.         0.         0.         0.          0.         0.         0.         0.99999443]
     [0.         0.         0.         0.         0.99999023 1.          0.         0.         0.         0.        ]
     [0.         0.         0.         0.99999023 0.         0.          0.         0.         0.         0.        ]
     [0.         0.         0.         1.         0.         0.          0.99999344 0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.99999344  0.         0.         0.         0.        ]
     [0.         0.         0.         0.         0.         0.          0.         0.         1.         0.        ]
     [0.         0.         0.         0.         0.         0.          0.         1.         0.         0.        ]
     [0.         0.         0.99999443 0.         0.         0.          0.         0.         0.         0.        ]]
     */
    assertArrayEquals(new double[][]{
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    }, m.toArray());
  }

  public void testFastIntersection() throws IOException {
    final CooMatrix distances = new IrisData(true).getDistances().toCoo();
    final float[] target = new float[distances.rows()];
    for (int k = 0; k < target.length; ++k) {
      target[k] = k % 3;
    }
    distances.fastIntersection(target, 1.0F, 1.0e8F);
    // Comparison values from Python
    assertArrayEquals(new double[][]{
      {0, 0, 0, 4.003748, 0, 0, 4.853864, 0, 0, 6.3450766},
      {0, 0, 0, 0, 3.6864617, 0, 0, 4.134005, 0, 0},
      {0, 0, 0, 0, 0, 4.4158807, 0, 0, 4.544227, 0},
      {4.003748, 0, 0, 0, 0, 0, 1.1, 0, 0, 9.126335},
      {0, 3.6864617, 0, 0, 0, 0, 0, 1.2165527, 0, 0},
      {0, 0, 4.4158807, 0, 0, 0, 0, 0, 1.4662877, 0},
      {4.853864, 0, 0, 1.1, 0, 0, 0, 0, 0, 9.481561},
      {0, 4.134005, 0, 0, 1.2165527, 0, 0, 0, 0, 0},
      {0, 0, 4.544227, 0, 0, 1.4662877, 0, 0, 0, 0},
      {6.3450766, 0, 0, 9.126335, 0, 0, 9.481561, 0, 0, 0},
    }, distances.toArray());
  }

  public void testMakeEpochsPerSample() {
    assertEquals("[84.0, 42.0, 10.5, 1.0]", Arrays.toString(Umap.makeEpochsPerSample(new float[] {0.5F, 1, 4, 42}, 10)));
  }

  /** A weight of zero means "never sample", which is marked with a negative value. */
  public void testMakeEpochsPerSampleMarksZeroWeights() {
    final float[] result = Umap.makeEpochsPerSample(new float[] {0, 4, 0, 42}, 10);
    assertTrue("zero weight not marked: " + result[0], result[0] < 0);
    assertTrue("zero weight not marked: " + result[2], result[2] < 0);
    assertEquals(10.5F, result[1], 1e-6);
    assertEquals(1.0F, result[3], 1e-6);
  }

  /**
   * A 1-simplex marked as "never sample" must not be optimized at all. The marker is
   * negative, so a due date carried straight into the loop would already be in the past in
   * epoch zero and would drift further into the past as it is advanced -- the simplex would
   * be sampled in every epoch instead of in none, the exact opposite of what is meant.
   *
   * Vertices 2 and 3 appear only in the marked simplex. Negative sampling reads other
   * vertices but only ever writes the head of the simplex it is processing, so if the marked
   * simplex is skipped those two rows must come back exactly as they went in.
   */
  public void testOptimizeLayoutSkipsUnsampledSimplices() {
    final float[][] initial = {{1, 2}, {3, 4}, {-5, -6}, {-7, -8}};
    final float[][] positions = new float[initial.length][];
    for (int i = 0; i < initial.length; ++i) {
      positions[i] = Arrays.copyOf(initial[i], initial[i].length);
    }
    final DefaultMatrix embedding = new DefaultMatrix(positions);

    final int[] head = {0, 2};
    final int[] tail = {1, 3};
    final float[] epochsPerSample = {1.0F, -1.0F};   // second simplex is never to be sampled

    new Umap().optimizeLayout(embedding, embedding, head, tail, 20, initial.length, epochsPerSample,
      1.577F, 0.895F, new Random(42), 1.0F, 1.0F, 5.0F, 1, false);

    assertTrue("vertex 0 was not optimized at all", positions[0][0] != initial[0][0] || positions[0][1] != initial[0][1]);
    for (int i = 2; i < initial.length; ++i) {
      assertEquals("vertex " + i + " moved", initial[i][0], positions[i][0]);
      assertEquals("vertex " + i + " moved", initial[i][1], positions[i][1]);
    }
  }

  /**
   * The same guarantee as the test above, but on a graph big enough that the optimization
   * actually splits across workers. The chunk boundaries must not disturb the schedule state:
   * epochOfNextSample is indexed per 1-simplex, so each entry has exactly one writer, and a
   * marked simplex has to stay unsampled no matter which chunk it lands in.
   *
   * The 1-simplices are disjoint pairs, so a vertex can only be moved by its own simplex.
   * Negative sampling reads foreign vertices but writes only the head of the simplex it is
   * processing, which is why the marked vertices are untouched even though other threads are
   * concurrently sampling them.
   */
  public void testOptimizeLayoutParallelSkipsUnsampledSimplices() {
    final int nEdges = 2 * Umap.MIN_EDGES_PER_WORKER;   // enough to be given two workers
    final int nVertices = 2 * nEdges;
    final int[] head = new int[nEdges];
    final int[] tail = new int[nEdges];
    final float[] epochsPerSample = new float[nEdges];
    final float[][] positions = new float[nVertices][2];
    final Random setup = new Random(7);
    for (int i = 0; i < nEdges; ++i) {
      head[i] = 2 * i;
      tail[i] = 2 * i + 1;
      epochsPerSample[i] = i % 100 == 0 ? -1.0F : 1.0F;   // every hundredth is never to be sampled
    }
    for (final float[] row : positions) {
      row[0] = setup.nextFloat() * 20 - 10;
      row[1] = setup.nextFloat() * 20 - 10;
    }
    final float[][] initial = new float[nVertices][];
    for (int i = 0; i < nVertices; ++i) {
      initial[i] = Arrays.copyOf(positions[i], 2);
    }
    final DefaultMatrix embedding = new DefaultMatrix(positions);

    new Umap().optimizeLayout(embedding, embedding, head, tail, 10, nVertices, epochsPerSample,
      1.577F, 0.895F, new Random(42), 1.0F, 1.0F, 5.0F, 2, false);

    int moved = 0;
    for (int i = 0; i < nEdges; ++i) {
      for (final int v : new int[] {head[i], tail[i]}) {
        final boolean same = initial[v][0] == positions[v][0] && initial[v][1] == positions[v][1];
        if (epochsPerSample[i] < 0) {
          assertTrue("unsampled vertex " + v + " moved", same);
        } else if (!same) {
          ++moved;
        }
      }
    }
    assertTrue("nothing was optimized at all", moved > nEdges);
  }

  // The fixture below is deliberately small and fully specified so that the exact result of
  // the single threaded optimization can be pinned. Keep it in step with GOLDEN_SERIAL_LAYOUT.
  private static final float[][] LAYOUT_START = {
    {-3.5F, 2.0F}, {1.25F, -0.75F}, {4.0F, 4.0F}, {-1.5F, -2.25F},
    {0.5F, 3.25F}, {2.75F, -4.0F}, {-4.25F, -0.5F}, {3.0F, 1.5F},
  };
  private static final int[] LAYOUT_HEAD = {0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 5};
  private static final int[] LAYOUT_TAIL = {1, 2, 3, 4, 5, 6, 7, 0, 4, 6, 7, 3};
  private static final float[] LAYOUT_EPOCHS_PER_SAMPLE = {
    1.0F, 2.0F, 1.5F, 3.0F, 1.0F, 5.0F, 2.5F, 1.25F, 4.0F, 1.0F, 7.0F, 2.0F,
  };

  /**
   * What single threaded optimizeLayout produced on 2026-08-17, before the layout SGD gained
   * a parallel path. Unlike the embedding sums this file used to carry, this one is not a
   * proxy for quality: its only job is to hold the serial path still while the method is
   * restructured around it, so a change here is a genuine change in behaviour and should be
   * regenerated only deliberately.
   */
  private static final float[][] GOLDEN_SERIAL_LAYOUT = {
    {-0.08309409F, 1.391967F},
    {1.033315F, 1.4856343F},
    {2.4101315F, 2.864756F},
    {0.16156346F, -1.5068393F},
    {0.2535907F, 2.1569846F},
    {1.1128602F, -2.102322F},
    {-2.6119025F, 0.15524451F},
    {0.4672074F, 0.15103333F},
  };

  private static float[][] runLayoutFixture(final int threads, final Random random) {
    final float[][] positions = new float[LAYOUT_START.length][];
    for (int i = 0; i < LAYOUT_START.length; ++i) {
      positions[i] = Arrays.copyOf(LAYOUT_START[i], 2);
    }
    // One matrix for both ends, as simplicialSetEmbedding uses it, so moveOther is true.
    final DefaultMatrix embedding = new DefaultMatrix(positions);
    new Umap().optimizeLayout(embedding, embedding, LAYOUT_HEAD, LAYOUT_TAIL, 10, LAYOUT_START.length,
      Arrays.copyOf(LAYOUT_EPOCHS_PER_SAMPLE, LAYOUT_EPOCHS_PER_SAMPLE.length),
      1.577F, 0.895F, random, 1.0F, 1.0F, 5.0F, threads, false);
    return positions;
  }

  public void testOptimizeLayoutSingleThreadMatchesFrozenResult() {
    final float[][] actual = runLayoutFixture(1, new Random(42));
    for (int i = 0; i < GOLDEN_SERIAL_LAYOUT.length; ++i) {
      for (int d = 0; d < 2; ++d) {
        assertEquals("vertex " + i + " component " + d, GOLDEN_SERIAL_LAYOUT[i][d], actual[i][d], 0.0F);
      }
    }
  }

  public void testOptimizeLayoutSingleThreadIsReproducible() {
    final float[][] first = runLayoutFixture(1, new Random(1234));
    final float[][] second = runLayoutFixture(1, new Random(1234));
    for (int i = 0; i < first.length; ++i) {
      assertTrue("vertex " + i + " differs between runs", Arrays.equals(first[i], second[i]));
    }
  }

  /**
   * Records which thread asks it for numbers. Negative sampling is the only consumer of the
   * random source inside optimizeLayout, so this sees every worker that does any work.
   */
  @SuppressWarnings("serial")
  private static final class ThreadRecordingRandom extends Random {
    private final List<Thread> mCallers = new ArrayList<>();

    ThreadRecordingRandom(final long seed) {
      super(seed);
    }

    @Override
    public int nextInt(final int bound) {
      synchronized (mCallers) {
        mCallers.add(Thread.currentThread());
      }
      return super.nextInt(bound);
    }
  }

  /**
   * The single threaded path must not merely produce the same numbers, it must not hand the
   * work to anyone else: callers rely on threads = 1 meaning no pool is created and no thread
   * is started. Checking who draws the negative samples proves that directly, without
   * depending on timing or on counting live threads.
   */
  public void testOptimizeLayoutSingleThreadRunsOnTheCallingThread() {
    final ThreadRecordingRandom random = new ThreadRecordingRandom(42);
    runLayoutFixture(1, random);
    assertFalse("no negative sampling happened, the test proves nothing", random.mCallers.isEmpty());
    for (final Thread caller : random.mCallers) {
      assertSame("work was handed to another thread", Thread.currentThread(), caller);
    }
  }

  /**
   * The parallel path lets concurrent updates to the same embedding row overwrite one
   * another, so nothing about the result is exact. What must survive is the purpose of the
   * embedding, which is why this asserts on neighbour agreement rather than on values.
   */
  public void testParallelLayoutProducesUsableEmbedding() throws IOException {
    final Data data = new DigitData();
    final Umap umap = new Umap();
    umap.setThreads(4);
    final float[][] matrix = umap.fitTransform(data.getData());
    assertEquals(1797, matrix.length);
    assertEquals(2, matrix[0].length);
    for (final float[] row : matrix) {
      assertTrue("embedding is not finite", Float.isFinite(row[0]) && Float.isFinite(row[1]));
    }
    // Ten classes, so chance is 0.1. Measured over ten seeds: 0.9726 to 0.9784 at four
    // threads against 0.9708 to 0.9792 at one, so the races cost nothing detectable here.
    assertAgreementAtLeast(0.94, matrix, data.getSampleClassIndex());
  }

  /**
   * A matrix wide enough that {@code rows * rows * columns} clears
   * {@link PairwiseDistances#MIN_PARALLEL_WORK}. Without that, every call below would fall to
   * the serial path and the comparison would be against itself.
   */
  private static Matrix distanceFixture(final int rows) {
    final int cols = (int) (PairwiseDistances.MIN_PARALLEL_WORK / ((long) rows * rows)) + 1;
    final Random random = new Random(99);
    final float[][] data = new float[rows][cols];
    for (final float[] row : data) {
      for (int j = 0; j < cols; ++j) {
        row[j] = random.nextFloat();
      }
    }
    return new DefaultMatrix(data);
  }

  /**
   * Splitting the distance matrix over threads is not one of the parallel paths that trade
   * reproducibility for speed: every cell is computed by exactly one worker, so the result
   * must be bit identical however many workers there are. Raw bits rather than a tolerance,
   * because anything short of equality here is a defect.
   */
  public void testPairwiseDistancesParallelMatchesSerial() {
    final int n = 97;
    final Matrix x = distanceFixture(n);
    final Matrix serial = PairwiseDistances.pairwiseDistances(x, EuclideanMetric.SINGLETON);
    // None of these divides 97, so every worker ends on a ragged stride, and 16 exceeds the
    // number of rows a worker could usefully hold.
    for (final int threads : new int[] {2, 3, 4, 7, 16}) {
      final Matrix parallel = PairwiseDistances.pairwiseDistances(x, EuclideanMetric.SINGLETON, threads);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          assertEquals("threads=" + threads + " at [" + i + "][" + j + "]",
            Float.floatToRawIntBits(serial.get(i, j)), Float.floatToRawIntBits(parallel.get(i, j)));
        }
      }
    }
    // Two implementations that agreed on garbage would still pass the loop above.
    for (int i = 0; i < n; ++i) {
      assertEquals("diagonal", 0.0F, serial.get(i, i), 0.0F);
      for (int j = 0; j < i; ++j) {
        assertEquals("not symmetric", serial.get(i, j), serial.get(j, i), 0.0F);
        assertTrue("distinct points at zero distance", serial.get(i, j) > 0);
      }
    }
  }

  /** Records who computed each distance, so that a claim about threads can be checked directly. */
  private static final class ThreadRecordingMetric extends MinkowskiMetric {
    private final List<Thread> mCallers = new ArrayList<>();

    private ThreadRecordingMetric() {
      super(2.0);
    }

    @Override
    public float distance(final float[] x, final float[] y) {
      synchronized (mCallers) {
        mCallers.add(Thread.currentThread());
      }
      return super.distance(x, y);
    }
  }

  /**
   * The single threaded path must not merely produce the same numbers, it must not hand the
   * work to anyone else: callers rely on threads = 1 meaning no pool is created and no thread
   * is started. The same holds for a matrix too small to repay a pool, whatever is asked for.
   * Checking who computed the distances proves both directly, without depending on timing or
   * on counting live threads.
   */
  public void testPairwiseDistancesSingleThreadRunsOnTheCallingThread() {
    final ThreadRecordingMetric metric = new ThreadRecordingMetric();
    PairwiseDistances.pairwiseDistances(distanceFixture(97), metric, 1);
    assertFalse("no distance was computed, the test proves nothing", metric.mCallers.isEmpty());
    for (final Thread caller : metric.mCallers) {
      assertSame("work was handed to another thread", Thread.currentThread(), caller);
    }

    final ThreadRecordingMetric belowThreshold = new ThreadRecordingMetric();
    PairwiseDistances.pairwiseDistances(new DefaultMatrix(new float[32][8]), belowThreshold, 6);
    assertFalse("no distance was computed, the test proves nothing", belowThreshold.mCallers.isEmpty());
    for (final Thread caller : belowThreshold.mCallers) {
      assertSame("a pool was started for a matrix too small to repay it", Thread.currentThread(), caller);
    }
  }

  /**
   * A pair of matrices wide enough that {@code xRows * yRows * columns} clears
   * {@link PairwiseDistances#MIN_PARALLEL_WORK}. Without that, every call below would fall to
   * the serial path and the comparison would be against itself.
   */
  private static Matrix[] rectangularDistanceFixture(final int xn, final int yn) {
    final int cols = (int) (PairwiseDistances.MIN_PARALLEL_WORK / ((long) xn * yn)) + 1;
    final Random random = new Random(1234);
    final float[][] x = new float[xn][cols];
    final float[][] y = new float[yn][cols];
    for (final float[][] data : new float[][][] {x, y}) {
      for (final float[] row : data) {
        for (int j = 0; j < cols; ++j) {
          row[j] = random.nextFloat();
        }
      }
    }
    return new Matrix[] {new DefaultMatrix(x), new DefaultMatrix(y)};
  }

  /**
   * The rectangular distance matrix is split the same way and owes the same guarantee as the
   * square one: every cell is computed by exactly one worker, so the result must be bit
   * identical however many workers there are.
   */
  public void testRectangularPairwiseDistancesParallelMatchesSerial() {
    final int xn = 41;
    final int yn = 97;
    final Matrix[] fixture = rectangularDistanceFixture(xn, yn);
    final Matrix x = fixture[0];
    final Matrix y = fixture[1];
    final Matrix serial = PairwiseDistances.pairwiseDistances(x, y, EuclideanMetric.SINGLETON);
    // None of these divides 41, so every worker ends on a ragged block, and 16 exceeds the
    // number of rows a worker could usefully hold.
    for (final int threads : new int[] {2, 3, 4, 7, 16}) {
      final Matrix parallel = PairwiseDistances.pairwiseDistances(x, y, EuclideanMetric.SINGLETON, threads);
      assertEquals(xn, parallel.rows());
      assertEquals(yn, parallel.cols());
      for (int i = 0; i < xn; ++i) {
        for (int j = 0; j < yn; ++j) {
          assertEquals("threads=" + threads + " at [" + i + "][" + j + "]",
            Float.floatToRawIntBits(serial.get(i, j)), Float.floatToRawIntBits(parallel.get(i, j)));
        }
      }
    }
  }

  /**
   * Two implementations agreeing on garbage would still pass the comparison above, and a
   * rectangular matrix has neither the symmetry nor the zero diagonal that the square test
   * checks itself against. The square overload is an independent implementation of the same
   * function, so measuring a matrix against itself must reproduce it cell for cell. This is
   * what would catch a transposed index, which is the mistake this shape invites.
   */
  public void testRectangularPairwiseDistancesAgreesWithTheSquareOverload() {
    final int n = 97;
    final Matrix x = distanceFixture(n);
    final Matrix square = PairwiseDistances.pairwiseDistances(x, EuclideanMetric.SINGLETON);
    for (final int threads : new int[] {1, 3, 7}) {
      final Matrix rectangular = PairwiseDistances.pairwiseDistances(x, x, EuclideanMetric.SINGLETON, threads);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          assertEquals("threads=" + threads + " at [" + i + "][" + j + "]",
            Float.floatToRawIntBits(square.get(i, j)), Float.floatToRawIntBits(rectangular.get(i, j)));
        }
      }
    }
  }

  /**
   * As for the square overload, the single threaded path must not merely produce the same
   * numbers, it must not hand the work to anyone else. The floor changes no value at all, so
   * this is the only test that can see it.
   */
  public void testRectangularPairwiseDistancesSingleThreadRunsOnTheCallingThread() {
    final Matrix[] fixture = rectangularDistanceFixture(41, 97);
    final ThreadRecordingMetric metric = new ThreadRecordingMetric();
    PairwiseDistances.pairwiseDistances(fixture[0], fixture[1], metric, 1);
    assertFalse("no distance was computed, the test proves nothing", metric.mCallers.isEmpty());
    for (final Thread caller : metric.mCallers) {
      assertSame("work was handed to another thread", Thread.currentThread(), caller);
    }

    final ThreadRecordingMetric belowThreshold = new ThreadRecordingMetric();
    PairwiseDistances.pairwiseDistances(new DefaultMatrix(new float[32][8]),
      new DefaultMatrix(new float[16][8]), belowThreshold, 6);
    assertFalse("no distance was computed, the test proves nothing", belowThreshold.mCallers.isEmpty());
    for (final Thread caller : belowThreshold.mCallers) {
      assertSame("a pool was started for a matrix too small to repay it", Thread.currentThread(), caller);
    }
  }

  /**
   * Transforming a single new instance clears the work threshold easily but has one row to
   * divide, and a pool cannot repay itself on it: measured at 0.85x against the serial loop.
   * The worker count is capped at the rows available, which sends this case to the calling
   * thread; that cap is the only thing protecting it, so it is asserted rather than assumed.
   */
  public void testRectangularPairwiseDistancesWithOneQueryRow() {
    final Matrix[] fixture = rectangularDistanceFixture(1, 97);
    assertTrue("fixture does not reach the parallel path",
      (long) fixture[0].cols() * fixture[1].rows() >= PairwiseDistances.MIN_PARALLEL_WORK);
    final ThreadRecordingMetric metric = new ThreadRecordingMetric();
    final Matrix distances = PairwiseDistances.pairwiseDistances(fixture[0], fixture[1], metric, 16);
    assertFalse("no distance was computed, the test proves nothing", metric.mCallers.isEmpty());
    for (final Thread caller : metric.mCallers) {
      assertSame("a pool was started for a single row", Thread.currentThread(), caller);
    }
    // The same metric, so that a difference here can only come from the split.
    final Matrix serial = PairwiseDistances.pairwiseDistances(fixture[0], fixture[1], new ThreadRecordingMetric());
    for (int j = 0; j < fixture[1].rows(); ++j) {
      assertEquals("at [0][" + j + "]", Float.floatToRawIntBits(serial.get(0, j)),
        Float.floatToRawIntBits(distances.get(0, j)));
    }
  }

  /** A deterministic graph in which every vertex has nNeighbors forward edges. */
  private static Heap candidateFixture(final int nVertices, final int nNeighbors) {
    final Heap graph = new Heap(nVertices, nNeighbors);
    final Random random = new Random(1234);
    for (int i = 0; i < nVertices; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        // Distinct targets per row, spread far enough apart that reverse edges reach across
        // any chunk boundary the parallel variant might draw.
        graph.push(i, random.nextFloat(), (i + 1 + j * 37) % nVertices, true);
      }
    }
    return graph;
  }

  private static List<Integer> sortedCandidates(final Heap heap, final int row, final int maxCandidates) {
    final List<Integer> present = new ArrayList<>();
    for (int j = 0; j < maxCandidates; ++j) {
      if (heap.index(row, j) >= 0) {
        present.add(heap.index(row, j));
      }
    }
    Collections.sort(present);
    return present;
  }

  /**
   * The parallel buildCandidates draws its priorities from per-chunk random streams, so the
   * heaps it produces are not element-for-element equal to the serial ones. What must hold is
   * that the same candidates are found: the random values only decide who is evicted once a
   * row overflows, and with maxCandidates well above the arrival count nothing overflows. Any
   * candidate lost to a race, or any reverse edge dropped at a chunk boundary, breaks this.
   */
  public void testParallelBuildCandidatesFindsTheSameCandidates() throws InterruptedException {
    final int nVertices = 500;
    final int nNeighbors = 6;
    final int maxCandidates = 60;   // comfortably above the ~12 arrivals per row
    final ExecutorService executor = Executors.newFixedThreadPool(4);
    try {
      final Heap serialGraph = candidateFixture(nVertices, nNeighbors);
      final Heap parallelGraph = candidateFixture(nVertices, nNeighbors);
      final Heap serial = serialGraph.buildCandidates(nVertices, nNeighbors, maxCandidates, new Random(7));
      final Heap parallel = parallelGraph.buildCandidates(nVertices, nNeighbors, maxCandidates,
        executor, Utils.splitRandom(new Random(7), 4));

      int nonEmpty = 0;
      for (int i = 0; i < nVertices; ++i) {
        final List<Integer> expected = sortedCandidates(serial, i, maxCandidates);
        assertEquals("row " + i, expected, sortedCandidates(parallel, i, maxCandidates));
        if (!expected.isEmpty()) {
          ++nonEmpty;
        }
      }
      assertEquals("fixture produced no candidates, the test proves nothing", nVertices, nonEmpty);

      // Both variants must also have consumed the source graph's new flags.
      for (int i = 0; i < nVertices; ++i) {
        for (int j = 0; j < nNeighbors; ++j) {
          assertFalse("row " + i + " column " + j, parallelGraph.isNew(i, j));
        }
      }
    } finally {
      executor.shutdown();
    }
  }

  /**
   * Covers ParallelNearestNeighborDescent itself, which no other test reaches: the layout
   * tests all run on data below SMALL_PROBLEM_THRESHOLD, where the neighbour search is
   * exhaustive and this class is never constructed. Approximate descent gives no exact
   * guarantee, so this asserts that the parallel run finds neighbours of the same quality as
   * the serial one rather than the identical ones.
   */
  public void testParallelNearestNeighborDescentFindsComparableNeighbours() throws IOException {
    final Matrix instances = new DefaultMatrix(new DigitData().getData());
    final int k = 15;
    final IndexedDistances serial = Umap.nearestNeighbors(instances, k, EuclideanMetric.SINGLETON, false, new Random(42), 1, false);
    final IndexedDistances parallel = Umap.nearestNeighbors(instances, k, EuclideanMetric.SINGLETON, false, new Random(42), 4, false);

    double serialSum = 0;
    double parallelSum = 0;
    for (int i = 0; i < instances.rows(); ++i) {
      for (int j = 0; j < k; ++j) {
        assertTrue("row " + i + " has an unfilled neighbour slot", parallel.getIndices()[i][j] >= 0);
        serialSum += serial.getDistances()[i][j];
        parallelSum += parallel.getDistances()[i][j];
      }
    }
    // Both are approximations of the same true kNN, so their total distance is the quality
    // measure. Measured over ten seeds: 0.99997 to 1.00005, i.e. the parallel descent is not
    // merely comparable but indistinguishable here. The bound is left wide all the same,
    // because it is meant to catch a descent that has stopped working, not to pin a number
    // that thread scheduling is entitled to move.
    final double ratio = parallelSum / serialSum;
    assertTrue("parallel descent found much worse neighbours: ratio " + ratio, ratio < 1.05);
    assertTrue("parallel descent found implausibly better neighbours: ratio " + ratio, ratio > 0.95);
  }

  public void testClip() {
    assertEquals(0.0F, Umap.clip(0F));
    assertEquals(1.5F, Umap.clip(1.5F));
    assertEquals(4.0F, Umap.clip(4));
    assertEquals(4.0F, Umap.clip(4.01F));
    assertEquals(4.0F, Umap.clip(Float.POSITIVE_INFINITY));
    assertEquals(-4.0F, Umap.clip(-4));
    assertEquals(-4.0F, Umap.clip(-4.01F));
    assertEquals(-4.0F, Umap.clip(Float.NEGATIVE_INFINITY));
  }

  public void testNaNQuery() {
    final float[][] data = new float[5000][3];
    final Umap umap = new Umap();
    data[0][0] = Float.NaN;
    try {
      umap.fitTransform(data);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
    data[0][0] = Float.POSITIVE_INFINITY;
    try {
      umap.fitTransform(data);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
    data[0][0] = Float.NEGATIVE_INFINITY;
    try {
      umap.fitTransform(data);
      fail();
    } catch (final IllegalArgumentException e) {
      // expected
    }
  }
}
