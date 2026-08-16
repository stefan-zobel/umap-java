package tagbio.umap.fx;

import junit.framework.TestCase;

/**
 * The rule behind the <code>umap.threads</code> property. The core count is passed in rather
 * than read from the runtime, so these assertions hold on any machine.
 */
public class ThreadResolutionTest extends TestCase {

  private static final int CORES = 12;

  public void testUnsetMeansOneThread() {
    assertEquals(1, UmapViewer.resolveThreads(0, CORES));
  }

  public void testNegativeMeansOneThread() {
    assertEquals(1, UmapViewer.resolveThreads(-1, CORES));
    assertEquals(1, UmapViewer.resolveThreads(Integer.MIN_VALUE, CORES));
  }

  public void testARequestBelowTheCoreCountIsHonoured() {
    assertEquals(1, UmapViewer.resolveThreads(1, CORES));
    assertEquals(2, UmapViewer.resolveThreads(2, CORES));
    assertEquals(CORES - 1, UmapViewer.resolveThreads(CORES - 1, CORES));
  }

  /** The core count is the cap, so asking for more does not oversubscribe the machine. */
  public void testARequestAboveTheCoreCountIsCappedAtIt() {
    assertEquals(CORES, UmapViewer.resolveThreads(CORES + 1, CORES));
    assertEquals(CORES, UmapViewer.resolveThreads(64, CORES));
    assertEquals(CORES, UmapViewer.resolveThreads(Integer.MAX_VALUE, CORES));
  }

  /** One thread can be asked for explicitly, and is what leaving the property alone gives. */
  public void testOneThreadIsReachableBothWays() {
    assertEquals(1, UmapViewer.resolveThreads(1, CORES));
    assertEquals(1, UmapViewer.resolveThreads(0, CORES));
  }

  public void testARequestEqualToTheCoreCountIsHonoured() {
    assertEquals(CORES, UmapViewer.resolveThreads(CORES, CORES));
  }

  /** Umap is handed this value directly, so it may never be zero whatever it is told. */
  public void testNeverBelowOne() {
    for (int requested = -3; requested <= 3; ++requested) {
      for (int cores = 0; cores <= 3; ++cores) {
        final int threads = UmapViewer.resolveThreads(requested, cores);
        assertTrue("requested " + requested + " on " + cores + " cores gave " + threads, threads >= 1);
      }
    }
  }

  public void testAnAbsentPropertyReadsAsZero() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.clearProperty(UmapViewer.THREADS_PROPERTY);
    try {
      assertEquals(0, UmapViewer.requestedThreads());
    } finally {
      restore(previous);
    }
  }

  /** A typo in the POM must not stop the viewer starting; it falls back to one thread. */
  public void testAMalformedPropertyReadsAsZero() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "lots");
    try {
      assertEquals(0, UmapViewer.requestedThreads());
      assertEquals(1, UmapViewer.resolveThreads(UmapViewer.requestedThreads(), CORES));
    } finally {
      restore(previous);
    }
  }

  public void testAValidPropertyIsRead() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "8");
    try {
      assertEquals(8, UmapViewer.requestedThreads());
    } finally {
      restore(previous);
    }
  }

  /** The value the POM carries when the property has been left alone. */
  public void testThePomDefaultMeansOneThread() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "0");
    try {
      assertEquals(1, UmapViewer.resolveThreads(UmapViewer.requestedThreads(), CORES));
    } finally {
      restore(previous);
    }
  }

  private static void restore(final String previous) {
    if (previous == null) {
      System.clearProperty(UmapViewer.THREADS_PROPERTY);
    } else {
      System.setProperty(UmapViewer.THREADS_PROPERTY, previous);
    }
  }
}
