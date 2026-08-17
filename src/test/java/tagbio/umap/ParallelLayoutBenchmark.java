/*
 * BSD 3-Clause License
 * Copyright (c) 2017, Leland McInnes, 2019 Tag.bio (Java port).
 * See LICENSE.txt.
 */
package tagbio.umap;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import tagbio.umap.metric.EuclideanMetric;
import tagbio.umap.metric.Metric;
import tagbio.umap.metric.ReducedEuclideanMetric;

/**
 * Measures what the Hogwild style parallel SGD in {@code Umap.optimizeLayout} buys, and
 * whether the resulting embedding is still usable.
 *
 * Both runs go through the production method, once with {@code threads = 1} and once with
 * {@code threads = N}, on the identical fuzzy simplicial set and the identical initial
 * embedding; the comparison therefore isolates the SGD itself and measures the shipped code
 * rather than a copy of it.
 *
 * Not a JUnit test; run its {@code main}. See PARALLELIZATION-ANALYSIS.md for the measured
 * speedup and for why the quality figure it prints must be read narrowly.
 *
 * Usage: ParallelLayoutBenchmark [n] [d] [k] [threads] [epochs] [clusters] [seed]
 * System properties: benchmark.quality
 */
public final class ParallelLayoutBenchmark {

  private ParallelLayoutBenchmark() { }

  private static final int NEIGHBOR_POOL = 60;
  private static final int LOCAL_CONNECTIVITY = 1;
  private static final int N_COMPONENTS = 2;
  private static final float NEGATIVE_SAMPLE_RATE = 5.0F;
  private static final float REPULSION_STRENGTH = 1.0F;
  private static final float LEARNING_RATE = 1.0F;
  private static final int QUALITY_SAMPLE = 300;
  private static final int QUALITY_K = 15;

  private static float[][] makeData(final int n, final int d, final int clusters, final long seed, final int[] labels) {
    final Random random = new Random(seed);
    final float[][] centers = new float[clusters][d];
    for (final float[] center : centers) {
      for (int j = 0; j < d; ++j) {
        center[j] = random.nextFloat() < 0.2F ? random.nextFloat() : 0.0F;
      }
    }
    final float[][] data = new float[n][d];
    for (int i = 0; i < n; ++i) {
      final int label = random.nextInt(clusters);
      labels[i] = label;
      final float[] center = centers[label];
      final float[] row = data[i];
      for (int j = 0; j < d; ++j) {
        final float v = center[j] + 0.15F * (float) random.nextGaussian();
        row[j] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }
    return data;
  }

  /**
   * Fraction of each sampled point's k nearest embedding neighbors that come from the same
   * generating cluster. For blob data this is the meaningful quality signal: a working
   * embedding scores near 1.0, a destroyed one drops towards 1/clusters.
   *
   * The alternative, checking how many true high dimensional neighbors survive, is useless
   * here: with 784 noisy dimensions the within-cluster neighbor ranking is essentially
   * arbitrary, so even a perfect embedding cannot reproduce it.
   */
  private static double clusterPurity(final Matrix embedding, final int[] sample, final int[] labels) {
    final int n = embedding.rows();
    int hits = 0;
    int total = 0;
    for (final int i : sample) {
      final float[] ei = embedding.row(i);
      final int[] neighbors = topK(idx -> idx == i ? Float.MAX_VALUE
        : ReducedEuclideanMetric.SINGLETON.distance(ei, embedding.row(idx)), n);
      for (final int nb : neighbors) {
        if (nb >= 0) {
          if (labels[nb] == labels[i]) {
            ++hits;
          }
          ++total;
        }
      }
    }
    return (double) hits / total;
  }

  private static Matrix layout(final Matrix embedding, final int[] head, final int[] tail,
                               final int nEpochs, final int nVertices, final float[] epochsPerSample,
                               final float a, final float b, final Random random, final int threads) {
    return new Umap().optimizeLayout(embedding, embedding, head, tail, nEpochs, nVertices,
      epochsPerSample, a, b, random, REPULSION_STRENGTH, LEARNING_RATE, NEGATIVE_SAMPLE_RATE, threads, false);
  }

  private interface DistanceAt {
    float at(int index);
  }

  private static int[] topK(final DistanceAt f, final int n) {
    final float[] best = new float[QUALITY_K];
    final int[] bestIdx = new int[QUALITY_K];
    Arrays.fill(best, Float.MAX_VALUE);
    Arrays.fill(bestIdx, -1);
    for (int idx = 0; idx < n; ++idx) {
      final float d = f.at(idx);
      if (d < best[QUALITY_K - 1]) {
        int p = QUALITY_K - 1;
        while (p > 0 && best[p - 1] > d) {
          best[p] = best[p - 1];
          bestIdx[p] = bestIdx[p - 1];
          --p;
        }
        best[p] = d;
        bestIdx[p] = idx;
      }
    }
    return bestIdx;
  }

  public static void main(final String[] args) {
    final int n = args.length > 0 ? Integer.parseInt(args[0]) : 20000;
    final int d = args.length > 1 ? Integer.parseInt(args[1]) : 784;
    final int k = args.length > 2 ? Integer.parseInt(args[2]) : 15;
    final int threads = args.length > 3 ? Integer.parseInt(args[3]) : 6;
    final int epochsArg = args.length > 4 ? Integer.parseInt(args[4]) : 0;
    final int clusters = args.length > 5 ? Integer.parseInt(args[5]) : 10;
    final long seed = args.length > 6 ? Long.parseLong(args[6]) : 42L;
    final boolean quality = Boolean.parseBoolean(System.getProperty("benchmark.quality", "true"));

    System.out.println("cores=" + Runtime.getRuntime().availableProcessors()
      + " n=" + n + " d=" + d + " k=" + k + " threads=" + threads + " clusters=" + clusters);

    UmapProgress.reset(Integer.MAX_VALUE / 4);
    final int[] labels = new int[n];
    final float[][] raw = makeData(n, d, clusters, seed, labels);
    final Matrix instances = new DefaultMatrix(raw);
    final Metric metric = EuclideanMetric.SINGLETON;
    final Random random = new Random(seed);

    final float[] ab = Curve.curveFit(1.0F, 0.1F);
    final float a = ab[0];
    final float b = ab[1];

    // Build the graph exactly as Umap.fit does for the large-data path.
    final int nTrees = 5 + (int) (Math.round(Math.pow(instances.rows(), 0.5) / 20.0));
    final int nIters = Math.max(5, (int) (Math.round(MathUtils.log2(instances.rows()))));
    final List<FlatTree> forest = RandomProjectionTree.makeForest(instances, k, nTrees, random, false, threads);
    final NearestNeighborDescent nnd = new ParallelNearestNeighborDescent(metric, threads);
    final Heap nn = nnd.descent(instances, k, random, NEIGHBOR_POOL, true, nIters, forest);
    final float[][] sigmasRhos = Umap.smoothKnnDist(nn.weights(), k, LOCAL_CONNECTIVITY);
    final Matrix membership = Umap.computeMembershipStrengths(nn.indices(), nn.weights(),
      sigmasRhos[0], sigmasRhos[1], instances.rows(), instances.rows()).eliminateZeros();
    final Matrix prod = membership.hadamardMultiplyTranspose();
    final CooMatrix graph = membership.addTranspose().subtract(prod).multiply(1.0F)
      .add(prod.multiply(0.0F)).eliminateZeros().toCoo();

    final int nVertices = graph.cols();
    final int nEpochs = epochsArg > 0 ? epochsArg : (graph.rows() <= 10000 ? 500 : 200);
    final float[] epochsPerSample = Umap.makeEpochsPerSample(graph.data(), nEpochs);
    final int[] head = graph.row();
    final int[] tail = graph.col();
    System.out.println("edges=" + head.length + " epochs=" + nEpochs + " vertices=" + nVertices);

    // Identical starting point for both runs.
    final float[][] init = MathUtils.uniform(new Random(seed + 7), -10, 10, graph.rows(), N_COMPONENTS);
    final float[][] initA = new float[init.length][];
    final float[][] initB = new float[init.length][];
    for (int i = 0; i < init.length; ++i) {
      initA[i] = Arrays.copyOf(init[i], N_COMPONENTS);
      initB[i] = Arrays.copyOf(init[i], N_COMPONENTS);
    }

    // Short warmup of both code paths on a fraction of the epochs, each with moveOther active.
    final int warmEpochs = Math.max(2, nEpochs / 20);
    final DefaultMatrix warmSerial = new DefaultMatrix(new float[init.length][N_COMPONENTS]);
    layout(warmSerial, head, tail, warmEpochs, nVertices,
      Arrays.copyOf(epochsPerSample, epochsPerSample.length), a, b, new Random(1), 1);
    final DefaultMatrix warmParallel = new DefaultMatrix(new float[init.length][N_COMPONENTS]);
    layout(warmParallel, head, tail, warmEpochs, nVertices,
      Arrays.copyOf(epochsPerSample, epochsPerSample.length), a, b, new Random(1), threads);

    final DefaultMatrix embeddingSerial = new DefaultMatrix(initA);
    long t = System.nanoTime();
    layout(embeddingSerial, head, tail, nEpochs, nVertices,
      Arrays.copyOf(epochsPerSample, epochsPerSample.length), a, b, new Random(seed + 11), 1);
    final long serialNanos = System.nanoTime() - t;

    final DefaultMatrix embeddingParallel = new DefaultMatrix(initB);
    t = System.nanoTime();
    layout(embeddingParallel, head, tail, nEpochs, nVertices,
      Arrays.copyOf(epochsPerSample, epochsPerSample.length), a, b, new Random(seed + 11), threads);
    final long parallelNanos = System.nanoTime() - t;

    System.out.println();
    System.out.printf("optimizeLayout threads=1           : %9.1f ms%n", serialNanos / 1e6);
    System.out.printf("optimizeLayout threads=%-2d          : %9.1f ms%n", threads, parallelNanos / 1e6);
    System.out.printf("speedup                            : %9.2fx%n", (double) serialNanos / parallelNanos);
    System.out.printf("embedding finite (serial/parallel) : %s / %s%n",
      embeddingSerial.isFinite(), embeddingParallel.isFinite());

    if (quality) {
      final Random qr = new Random(seed + 3);
      final int[] sample = new int[Math.min(QUALITY_SAMPLE, n)];
      for (int s = 0; s < sample.length; ++s) {
        sample[s] = qr.nextInt(n);
      }
      System.out.printf("cluster purity serial              : %9.4f%n",
        clusterPurity(embeddingSerial, sample, labels));
      System.out.printf("cluster purity parallel            : %9.4f%n",
        clusterPurity(embeddingParallel, sample, labels));
      System.out.printf("  (k=%d, %d sampled points, 1.0 = perfect, %.3f = random)%n",
        QUALITY_K, sample.length, 1.0 / clusters);
    }
  }
}
