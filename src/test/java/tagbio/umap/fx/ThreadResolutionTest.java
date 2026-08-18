package tagbio.umap.fx;

import java.util.List;

import junit.framework.TestCase;

/**
 * The rule behind the <code>umap.threads</code> property and the range the thread dropdown
 * offers. The core count is passed in rather than read from the runtime, so these assertions
 * hold on any machine.
 */
public class ThreadResolutionTest extends TestCase {

  private static final int CORES = 12;

  /**
   * Nothing configured is not a request for one thread, it is no request at all, and is
   * answered with half the machine.
   */
  public void testNothingConfiguredMeansHalfTheCores() {
    assertEquals(6, UmapViewer.resolveThreads(null, CORES));
    assertEquals(4, UmapViewer.resolveThreads(null, 8));
    assertEquals(2, UmapViewer.resolveThreads(null, 4));
  }

  /** Half of an odd count rounds down, and half of one or two is still one. */
  public void testHalfTheCoresNeverFallsBelowOne() {
    assertEquals(3, UmapViewer.resolveThreads(null, 7));
    assertEquals(2, UmapViewer.resolveThreads(null, 5));
    assertEquals(1, UmapViewer.resolveThreads(null, 3));
    assertEquals(1, UmapViewer.resolveThreads(null, 2));
    assertEquals(1, UmapViewer.resolveThreads(null, 1));
    assertEquals(1, UmapViewer.resolveThreads(null, 0));
  }

  /** Configured at or below zero keeps the rule it always had: one thread. */
  public void testAConfiguredZeroOrLessMeansOneThread() {
    assertEquals(1, UmapViewer.resolveThreads(0, CORES));
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

  /**
   * One thread is still reachable, but now only by asking for it. This is the difference the
   * new default makes, and the one that would break a user who relied on an empty property
   * meaning a reproducible embedding.
   */
  public void testOneThreadHasToBeAskedForNow() {
    assertEquals(1, UmapViewer.resolveThreads(1, CORES));
    assertEquals(1, UmapViewer.resolveThreads(0, CORES));
    assertTrue(UmapViewer.resolveThreads(null, CORES) > 1);
  }

  public void testARequestEqualToTheCoreCountIsHonoured() {
    assertEquals(CORES, UmapViewer.resolveThreads(CORES, CORES));
  }

  /** Umap is handed this value directly, so it may never be zero whatever it is told. */
  public void testNeverBelowOne() {
    for (int cores = 0; cores <= 3; ++cores) {
      assertTrue("nothing configured on " + cores + " cores",
        UmapViewer.resolveThreads(null, cores) >= 1);
      for (int requested = -3; requested <= 3; ++requested) {
        final int threads = UmapViewer.resolveThreads(requested, cores);
        assertTrue("requested " + requested + " on " + cores + " cores gave " + threads, threads >= 1);
      }
    }
  }

  public void testTheChoicesRunFromOneToTheCoreCount() {
    final List<Integer> choices = UmapViewer.threadChoices(CORES);
    assertEquals(CORES, choices.size());
    assertEquals(Integer.valueOf(1), choices.get(0));
    assertEquals(Integer.valueOf(CORES), choices.get(CORES - 1));
    for (int i = 0; i < CORES; ++i) {
      assertEquals(Integer.valueOf(i + 1), choices.get(i));
    }
  }

  /** A single core machine still gets a usable dropdown rather than an empty one. */
  public void testTheChoicesAreNeverEmpty() {
    assertEquals(1, UmapViewer.threadChoices(1).size());
    assertEquals(1, UmapViewer.threadChoices(0).size());
    assertEquals(1, UmapViewer.threadChoices(-4).size());
  }

  /**
   * What the two would look wrong together as: a starting value the dropdown does not offer
   * shows as a blank box, and the first fit would then run on whatever the fallback is rather
   * than on what is on screen.
   */
  public void testTheStartingValueIsAlwaysOneOfTheChoices() {
    for (int cores = 0; cores <= 16; ++cores) {
      final List<Integer> choices = UmapViewer.threadChoices(cores);
      assertTrue("nothing configured on " + cores + " cores",
        choices.contains(UmapViewer.resolveThreads(null, cores)));
      for (int requested = -2; requested <= 20; ++requested) {
        assertTrue("requested " + requested + " on " + cores + " cores",
          choices.contains(UmapViewer.resolveThreads(requested, cores)));
      }
    }
  }

  public void testAnAbsentPropertyReadsAsNothingConfigured() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.clearProperty(UmapViewer.THREADS_PROPERTY);
    try {
      assertNull(UmapViewer.requestedThreads());
    } finally {
      restore(previous);
    }
  }

  /**
   * The state the POM ships in: the element is there but empty, so the exec plugin passes
   * <code>-Dumap.threads=</code>. That has to read as nothing configured, or the shipped
   * default would be one thread rather than half the machine.
   */
  public void testAnEmptyPropertyReadsAsNothingConfigured() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "");
    try {
      assertNull(UmapViewer.requestedThreads());
      assertEquals(6, UmapViewer.resolveThreads(UmapViewer.requestedThreads(), CORES));
    } finally {
      restore(previous);
    }
  }

  /**
   * An unsubstituted property, which is what reaches the viewer if the element is deleted from
   * the POM altogether rather than left empty. Both spellings of "not configured" agree.
   */
  public void testAnUnsubstitutedPropertyReadsAsNothingConfigured() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "${umap.threads}");
    try {
      assertNull(UmapViewer.requestedThreads());
    } finally {
      restore(previous);
    }
  }

  /** A typo in the POM must not stop the viewer starting; it falls back to the default. */
  public void testAMalformedPropertyReadsAsNothingConfigured() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "lots");
    try {
      assertNull(UmapViewer.requestedThreads());
      assertEquals(6, UmapViewer.resolveThreads(UmapViewer.requestedThreads(), CORES));
    } finally {
      restore(previous);
    }
  }

  public void testAValidPropertyIsRead() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "8");
    try {
      assertEquals(Integer.valueOf(8), UmapViewer.requestedThreads());
      assertEquals(8, UmapViewer.resolveThreads(UmapViewer.requestedThreads(), CORES));
    } finally {
      restore(previous);
    }
  }

  /** An explicit zero is a configured value, and still means one thread. */
  public void testAnExplicitZeroIsConfigured() {
    final String previous = System.getProperty(UmapViewer.THREADS_PROPERTY);
    System.setProperty(UmapViewer.THREADS_PROPERTY, "0");
    try {
      assertEquals(Integer.valueOf(0), UmapViewer.requestedThreads());
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
