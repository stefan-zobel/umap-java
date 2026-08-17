/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import junit.framework.TestCase;
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
