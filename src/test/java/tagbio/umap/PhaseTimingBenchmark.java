/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import tagbio.umap.metric.EuclideanMetric;
import tagbio.umap.metric.Metric;

/**
 * Measures how the wall-clock time of a UMAP fit distributes over the phases of the
 * algorithm. Lives in package {@code tagbio.umap} so that it can call the package private
 * pipeline methods directly; the production code needs no instrumentation.
 *
 * The pipeline below mirrors {@link Umap#fit} step by step. Where {@code fit} calls a
 * package private method, this benchmark calls the very same method, so every timing is of
 * production code and nothing here duplicates it.
 *
 * Not a JUnit test; run its {@code main}. See PARALLELIZATION-ANALYSIS.md for the numbers
 * this produced and how to interpret them.
 *
 * Usage: PhaseTimingBenchmark [n] [d] [k] [threads] [epochs] [clusters] [seed]
 * System properties: benchmark.reference, benchmark.warmup
 */
public final class PhaseTimingBenchmark {

  private PhaseTimingBenchmark() { }

  private static final int NEIGHBOR_POOL = 60;   // maxCandidates, as hard coded in Umap.nearestNeighbors
  private static final int LOCAL_CONNECTIVITY = 1;
  private static final float SET_OP_MIX_RATIO = 1.0F;
  private static final int N_COMPONENTS = 2;
  private static final int NEGATIVE_SAMPLE_RATE = 5;
  private static final float REPULSION_STRENGTH = 1.0F;
  private static final float LEARNING_RATE = 1.0F;

  /** Ordered phase name to accumulated nanoseconds. */
  private final Map<String, Long> mTimes = new LinkedHashMap<>();
  private long mMark;

  private void start() {
    mMark = System.nanoTime();
  }

  private void stop(final String phase) {
    final long elapsed = System.nanoTime() - mMark;
    mTimes.merge(phase, elapsed, Long::sum);
    mMark = System.nanoTime();
  }

  /**
   * Build clustered high dimensional data. Values are kept in [0, 1] with a large fraction
   * of near-zero attributes, which is roughly how MNIST pixel data behaves.
   */
  private static float[][] makeData(final int n, final int d, final int clusters, final long seed) {
    final Random random = new Random(seed);
    final float[][] centers = new float[clusters][d];
    for (final float[] center : centers) {
      for (int j = 0; j < d; ++j) {
        // Most attributes stay dark, a minority carries the cluster signal.
        center[j] = random.nextFloat() < 0.2F ? random.nextFloat() : 0.0F;
      }
    }
    final float[][] data = new float[n][d];
    for (int i = 0; i < n; ++i) {
      final float[] center = centers[random.nextInt(clusters)];
      final float[] row = data[i];
      for (int j = 0; j < d; ++j) {
        final float v = center[j] + 0.15F * (float) random.nextGaussian();
        row[j] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }
    return data;
  }

  /**
   * Run the large-data fit pipeline (Umap.fit, rows >= SMALL_PROBLEM_THRESHOLD) with a timer
   * around every phase.
   */
  private Matrix runLargePipeline(final Matrix instances, final int nNeighbors, final int threads,
                                  final int nEpochsRequested, final long seed) {
    final Metric metric = EuclideanMetric.SINGLETON;
    final Random random = new Random(seed);
    UmapProgress.reset(Integer.MAX_VALUE / 4);

    start();
    final float[] ab = Curve.curveFit(1.0F, 0.1F);
    final float a = ab[0];
    final float b = ab[1];
    stop("findAbParams");

    // ---- nearest neighbors, replicating the body of Umap.nearestNeighbors ----
    final int nTrees = 5 + (int) (Math.round(Math.pow(instances.rows(), 0.5) / 20.0));
    final int nIters = Math.max(5, (int) (Math.round(MathUtils.log2(instances.rows()))));

    start();
    final List<FlatTree> rpForest =
      RandomProjectionTree.makeForest(instances, nNeighbors, nTrees, random, metric.isAngular(), threads);
    stop("rpForest");

    // The same choice Umap.nearestNeighbors makes, so the descent measured here is the one
    // that ships. Its internal split between the serial buildCandidates and the parallel work
    // is not observable from outside; it was measured once with a timer carrying copy of
    // ParallelNearestNeighborDescent and came to 1-2 % of the fit, recorded in
    // PARALLELIZATION-ANALYSIS.md. The copy was then dropped rather than kept in sync.
    final NearestNeighborDescent descent = threads == 1
      ? new NearestNeighborDescent(metric)
      : new ParallelNearestNeighborDescent(metric, threads);
    start();
    final Heap nn = descent.descent(instances, nNeighbors, random, NEIGHBOR_POOL, true, nIters, rpForest);
    stop("nndDescent");
    final int[][] knnIndices = nn.indices();
    final float[][] knnDists = nn.weights();

    // ---- fuzzy simplicial set, replicating the body of Umap.fuzzySimplicialSet ----
    start();
    final float[][] sigmasRhos = Umap.smoothKnnDist(knnDists, nNeighbors, LOCAL_CONNECTIVITY);
    stop("smoothKnnDist");
    final float[] sigmas = sigmasRhos[0];
    final float[] rhos = sigmasRhos[1];

    start();
    final CooMatrix membership =
      Umap.computeMembershipStrengths(knnIndices, knnDists, sigmas, rhos, instances.rows(), instances.rows());
    stop("computeMembershipStrengths");

    start();
    final Matrix result = membership.eliminateZeros();
    stop("graph.eliminateZeros1");

    start();
    final Matrix prodMatrix = result.hadamardMultiplyTranspose();
    stop("graph.hadamardMultiplyTranspose");

    start();
    final Matrix sum = result.addTranspose();
    stop("graph.addTranspose");

    start();
    final Matrix diff = sum.subtract(prodMatrix);
    stop("graph.subtract");

    start();
    final Matrix scaled = diff.multiply(SET_OP_MIX_RATIO);
    final Matrix scaledProd = prodMatrix.multiply(1.0F - SET_OP_MIX_RATIO);
    stop("graph.multiply");

    start();
    final Matrix combined = scaled.add(scaledProd);
    stop("graph.add");

    start();
    final Matrix graphMatrix = combined.eliminateZeros();
    stop("graph.eliminateZeros2");

    // ---- simplicial set embedding, replicating the body of Umap.simplicialSetEmbedding ----
    start();
    CooMatrix graph = graphMatrix.toCoo();
    final int nVertices = graph.cols();
    final int nEpochs = nEpochsRequested > 0 ? nEpochsRequested : (graph.rows() <= 10000 ? 500 : 200);

    final int nnzBefore = graph.row().length;
    final float[] graphData = graph.mutableData();
    MathUtils.zeroEntriesBelowLimit(graphData, MathUtils.max(graphData) / (float) nEpochs);
    graph = (CooMatrix) graph.eliminateZeros();
    final int nnzAfter = graph.row().length;

    final Matrix embedding = new DefaultMatrix(MathUtils.uniform(random, -10, 10, graph.rows(), N_COMPONENTS));
    final float[] epochsPerSample = Umap.makeEpochsPerSample(graph.data(), nEpochs);
    final int[] head = graph.row();
    final int[] tail = graph.col();
    stop("embeddingSetup");

    start();
    final Matrix embedded = invokeOptimizeLayout(embedding, head, tail, nEpochs, nVertices,
      epochsPerSample, a, b, random, REPULSION_STRENGTH, LEARNING_RATE, threads);
    stop("optimizeLayout");

    mTimes.put("#edges (nnz) into SGD", (long) nnzAfter);
    mTimes.put("#edges pruned by threshold", (long) (nnzBefore - nnzAfter));
    mTimes.put("#epochs", (long) nEpochs);
    return embedded;
  }

  /**
   * Run the small-data fit pipeline (Umap.fit, rows < SMALL_PROBLEM_THRESHOLD) with a timer
   * around every phase.
   */
  private void runSmallPipeline(final Matrix instances, final int nNeighbors, final int nEpochsRequested,
                                final int threads, final long seed) {
    final Metric metric = EuclideanMetric.SINGLETON;
    final Random random = new Random(seed);
    UmapProgress.reset(Integer.MAX_VALUE / 4);

    start();
    final float[] ab = Curve.curveFit(1.0F, 0.1F);
    final float a = ab[0];
    final float b = ab[1];
    stop("findAbParams");

    start();
    final Matrix dmat = PairwiseDistances.pairwiseDistances(instances, metric, threads);
    stop("pairwiseDistances");

    // Umap.nearestNeighbors with PrecomputedMetric: fastKnnIndices plus a gather.
    start();
    final int[][] knnIndices = Utils.fastKnnIndices(dmat, nNeighbors, threads);
    stop("fastKnnIndices");

    start();
    final float[][] knnDists = new float[knnIndices.length][nNeighbors];
    for (int i = 0; i < knnDists.length; ++i) {
      for (int j = 0; j < nNeighbors; ++j) {
        knnDists[i][j] = dmat.get(i, knnIndices[i][j]);
      }
    }
    stop("knnDistanceGather");

    start();
    final float[][] sigmasRhos = Umap.smoothKnnDist(knnDists, nNeighbors, LOCAL_CONNECTIVITY);
    stop("smoothKnnDist");

    start();
    final CooMatrix membership = Umap.computeMembershipStrengths(knnIndices, knnDists,
      sigmasRhos[0], sigmasRhos[1], instances.rows(), instances.rows());
    final Matrix result = membership.eliminateZeros();
    final Matrix prodMatrix = result.hadamardMultiplyTranspose();
    final Matrix graphMatrix = result.addTranspose().subtract(prodMatrix)
      .multiply(SET_OP_MIX_RATIO).add(prodMatrix.multiply(1.0F - SET_OP_MIX_RATIO)).eliminateZeros();
    stop("graphAssembly");

    start();
    CooMatrix graph = graphMatrix.toCoo();
    final int nVertices = graph.cols();
    final int nEpochs = nEpochsRequested > 0 ? nEpochsRequested : (graph.rows() <= 10000 ? 500 : 200);
    final float[] graphData = graph.mutableData();
    MathUtils.zeroEntriesBelowLimit(graphData, MathUtils.max(graphData) / (float) nEpochs);
    graph = (CooMatrix) graph.eliminateZeros();
    final Matrix embedding = new DefaultMatrix(MathUtils.uniform(random, -10, 10, graph.rows(), N_COMPONENTS));
    final float[] epochsPerSample = Umap.makeEpochsPerSample(graph.data(), nEpochs);
    final int[] head = graph.row();
    final int[] tail = graph.col();
    stop("embeddingSetup");

    start();
    invokeOptimizeLayout(embedding, head, tail, nEpochs, nVertices, epochsPerSample,
      a, b, random, REPULSION_STRENGTH, LEARNING_RATE, threads);
    stop("optimizeLayout");

    mTimes.put("#edges (nnz) into SGD", (long) head.length);
    mTimes.put("#epochs", (long) nEpochs);
  }

  /**
   * Runs the production SGD unmodified: every value it depends on is passed as an argument,
   * so nothing about the measured code differs from what {@code fit} executes.
   */
  private static Matrix invokeOptimizeLayout(final Matrix embedding, final int[] head, final int[] tail,
                                             final int nEpochs, final int nVertices, final float[] epochsPerSample,
                                             final float a, final float b, final Random random,
                                             final float gamma, final float initialAlpha, final int threads) {
    return new Umap().optimizeLayout(embedding, embedding, head, tail, nEpochs, nVertices,
      epochsPerSample, a, b, random, gamma, initialAlpha, NEGATIVE_SAMPLE_RATE, threads, false);
  }

  private void report(final String title, final long referenceNanos) {
    System.out.println();
    System.out.println("=== " + title + " ===");
    long total = 0;
    for (final Map.Entry<String, Long> e : mTimes.entrySet()) {
      if (!e.getKey().startsWith("#") && !e.getKey().startsWith("  ")) {
        total += e.getValue();
      }
    }
    System.out.printf("%-34s %12s %8s%n", "phase", "ms", "%");
    System.out.println("-".repeat(56));
    for (final Map.Entry<String, Long> e : mTimes.entrySet()) {
      final String key = e.getKey();
      if (key.startsWith("#")) {
        System.out.printf("%-34s %12d%n", key, e.getValue());
      } else {
        System.out.printf("%-34s %12.1f %7.2f%%%n", key, e.getValue() / 1e6, 100.0 * e.getValue() / total);
      }
    }
    System.out.println("-".repeat(56));
    System.out.printf("%-34s %12.1f %7.2f%%%n", "TOTAL (sum of phases)", total / 1e6, 100.0);
    if (referenceNanos > 0) {
      System.out.printf("%-34s %12.1f%n", "reference Umap.fitTransform", referenceNanos / 1e6);
      System.out.printf("%-34s %12.1f%%%n", "benchmark/reference ratio", 100.0 * total / referenceNanos);
    }
    System.out.println();
  }

  public static void main(final String[] args) {
    final int n = args.length > 0 ? Integer.parseInt(args[0]) : 20000;
    final int d = args.length > 1 ? Integer.parseInt(args[1]) : 784;
    final int k = args.length > 2 ? Integer.parseInt(args[2]) : 15;
    final int threads = args.length > 3 ? Integer.parseInt(args[3]) : 6;
    final int epochs = args.length > 4 ? Integer.parseInt(args[4]) : 0;
    final int clusters = args.length > 5 ? Integer.parseInt(args[5]) : 10;
    final long seed = args.length > 6 ? Long.parseLong(args[6]) : 42L;
    final boolean reference = Boolean.parseBoolean(System.getProperty("benchmark.reference", "false"));
    final boolean warmup = Boolean.parseBoolean(System.getProperty("benchmark.warmup", "true"));

    System.out.println("cores=" + Runtime.getRuntime().availableProcessors()
      + " n=" + n + " d=" + d + " k=" + k + " threads=" + threads
      + " epochs=" + (epochs > 0 ? epochs : "default") + " clusters=" + clusters + " seed=" + seed);

    final long dataStart = System.nanoTime();
    final float[][] raw = makeData(n, d, clusters, seed);
    System.out.printf("data generated in %.1f ms%n", (System.nanoTime() - dataStart) / 1e6);

    if (warmup) {
      // JIT warmup on a scaled down copy of the pipeline that is about to be measured.
      final int wn = n < 4096 ? Math.min(n, 1500) : Math.min(n, 5000);
      final float[][] wraw = makeData(wn, d, clusters, seed + 1);
      final long ws = System.nanoTime();
      if (n < 4096) {
        new PhaseTimingBenchmark().runSmallPipeline(new DefaultMatrix(wraw), k, 20, threads, seed);
      } else {
        new PhaseTimingBenchmark().runLargePipeline(new DefaultMatrix(wraw), k, threads, 20, seed);
      }
      System.out.printf("warmup (n=%d) done in %.1f ms%n", wn, (System.nanoTime() - ws) / 1e6);
    }

    long referenceNanos = 0;
    if (reference) {
      final Umap umap = new Umap();
      umap.setNumberComponents(N_COMPONENTS);
      umap.setNumberNearestNeighbours(k);
      umap.setThreads(threads);
      if (epochs > 0) {
        umap.setNumberEpochs(epochs);
      }
      final long rs = System.nanoTime();
      umap.fitTransform(new DefaultMatrix(raw), null);
      referenceNanos = System.nanoTime() - rs;
    }

    final PhaseTimingBenchmark benchmark = new PhaseTimingBenchmark();
    if (n < 4096) {
      benchmark.runSmallPipeline(new DefaultMatrix(raw), k, epochs, threads, seed);
      // The neighbour search on this path is exhaustive and serial; only optimizeLayout is threaded.
      benchmark.report("small-data path  n=" + n + " d=" + d + " threads=" + threads + " (SGD only)", referenceNanos);
    } else {
      benchmark.runLargePipeline(new DefaultMatrix(raw), k, threads, epochs, seed);
      benchmark.report("large-data path  n=" + n + " d=" + d + " threads=" + threads, referenceNanos);
    }
  }
}
