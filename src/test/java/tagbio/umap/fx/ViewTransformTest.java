package tagbio.umap.fx;

import junit.framework.TestCase;

public class ViewTransformTest extends TestCase {

  private static final double TOLERANCE = 1.0e-9;
  private static final double WIDTH = 800;
  private static final double HEIGHT = 600;
  private static final double PADDING = 24;

  private ViewTransform fitted() {
    final ViewTransform view = new ViewTransform();
    // Square data range, so the taller viewport dimension is the limiting one.
    view.fit(-3, 7, 2, 12, WIDTH, HEIGHT, PADDING);
    return view;
  }

  public void testNotValidBeforeFitting() {
    assertFalse(new ViewTransform().isValid());
  }

  public void testNotValidForEmptyViewport() {
    final ViewTransform view = new ViewTransform();
    view.fit(0, 1, 0, 1, 0, 0, PADDING);
    assertFalse(view.isValid());
  }

  public void testFitPlacesBoundsOnThePadding() {
    final ViewTransform view = fitted();
    // Limiting scale is (600 - 48) / 10 = 55.2, so the data spans 552 pixels and is
    // centred in both directions.
    assertTrue(view.isValid());
    assertEquals(124.0, view.toScreenX(-3), TOLERANCE);
    assertEquals(676.0, view.toScreenX(7), TOLERANCE);
    assertEquals(576.0, view.toScreenY(2), TOLERANCE);
    assertEquals(24.0, view.toScreenY(12), TOLERANCE);
  }

  public void testYAxisIsFlipped() {
    final ViewTransform view = fitted();
    assertTrue("smaller y must be further down the screen", view.toScreenY(2) > view.toScreenY(12));
  }

  public void testBothAxesShareOneScale() {
    final ViewTransform view = new ViewTransform();
    // Deliberately non square: an aspect preserving fit must still use a single scale.
    view.fit(0, 20, 0, 5, WIDTH, HEIGHT, PADDING);
    final double pixelsPerUnitX = view.toScreenX(1) - view.toScreenX(0);
    final double pixelsPerUnitY = view.toScreenY(0) - view.toScreenY(1);
    assertEquals(pixelsPerUnitX, pixelsPerUnitY, TOLERANCE);
  }

  public void testFitIsCentredOnTheShorterAxis() {
    final ViewTransform view = new ViewTransform();
    view.fit(0, 20, 0, 5, WIDTH, HEIGHT, PADDING);
    // Scale is limited by x: (800 - 48) / 20 = 37.6, so y spans 188 pixels of the 600
    // available and the gaps above and below must match.
    final double above = view.toScreenY(5);
    final double below = HEIGHT - view.toScreenY(0);
    assertEquals(above, below, TOLERANCE);
  }

  public void testPanShiftsByExactPixels() {
    final ViewTransform view = fitted();
    final double beforeX = view.toScreenX(0);
    final double beforeY = view.toScreenY(0);
    view.pan(25, -10);
    assertEquals(beforeX + 25, view.toScreenX(0), TOLERANCE);
    assertEquals(beforeY - 10, view.toScreenY(0), TOLERANCE);
  }

  public void testPanDoesNotChangeScale() {
    final ViewTransform view = fitted();
    final double before = view.toScreenX(1) - view.toScreenX(0);
    view.pan(120, -75);
    assertEquals(before, view.toScreenX(1) - view.toScreenX(0), TOLERANCE);
  }

  public void testZoomInKeepsThePointUnderTheCursor() {
    final ViewTransform view = fitted();
    final float dataX = 2.0f;
    final float dataY = 7.0f;
    final double cursorX = view.toScreenX(dataX);
    final double cursorY = view.toScreenY(dataY);
    view.zoomAt(cursorX, cursorY, 1.15);
    assertEquals(cursorX, view.toScreenX(dataX), 1.0e-6);
    assertEquals(cursorY, view.toScreenY(dataY), 1.0e-6);
  }

  public void testZoomOutKeepsThePointUnderTheCursor() {
    final ViewTransform view = fitted();
    final float dataX = -1.5f;
    final float dataY = 11.25f;
    final double cursorX = view.toScreenX(dataX);
    final double cursorY = view.toScreenY(dataY);
    view.zoomAt(cursorX, cursorY, 1 / 1.15);
    assertEquals(cursorX, view.toScreenX(dataX), 1.0e-6);
    assertEquals(cursorY, view.toScreenY(dataY), 1.0e-6);
  }

  public void testRepeatedZoomKeepsThePointUnderTheCursor() {
    final ViewTransform view = fitted();
    final float dataX = 4.5f;
    final float dataY = 3.25f;
    final double cursorX = view.toScreenX(dataX);
    final double cursorY = view.toScreenY(dataY);
    for (int i = 0; i < 25; ++i) {
      view.zoomAt(cursorX, cursorY, 1.15);
    }
    assertEquals(cursorX, view.toScreenX(dataX), 1.0e-4);
    assertEquals(cursorY, view.toScreenY(dataY), 1.0e-4);
  }

  public void testZoomActuallyMagnifies() {
    final ViewTransform view = fitted();
    final double before = view.toScreenX(1) - view.toScreenX(0);
    view.zoomAt(400, 300, 2.0);
    assertEquals(2 * before, view.toScreenX(1) - view.toScreenX(0), 1.0e-6);
  }

  public void testZoomInIsBounded() {
    final ViewTransform view = fitted();
    final double fitted = view.toScreenX(1) - view.toScreenX(0);
    for (int i = 0; i < 500; ++i) {
      view.zoomAt(400, 300, 1.5);
    }
    // MAX_ZOOM is 400 times the fitted scale.
    assertEquals(400 * fitted, view.toScreenX(1) - view.toScreenX(0), 1.0e-3);
  }

  public void testZoomOutIsBounded() {
    final ViewTransform view = fitted();
    final double fitted = view.toScreenX(1) - view.toScreenX(0);
    for (int i = 0; i < 500; ++i) {
      view.zoomAt(400, 300, 0.5);
    }
    // MIN_ZOOM is a quarter of the fitted scale.
    assertEquals(0.25 * fitted, view.toScreenX(1) - view.toScreenX(0), 1.0e-9);
  }

  public void testZoomBeforeFittingIsIgnored() {
    final ViewTransform view = new ViewTransform();
    view.zoomAt(100, 100, 2.0);
    assertFalse(view.isValid());
  }

  public void testDegenerateRangeStillFits() {
    final ViewTransform view = new ViewTransform();
    // Every point identical: must not divide by zero or produce a broken transform.
    view.fit(5, 5, 5, 5, WIDTH, HEIGHT, PADDING);
    assertTrue(view.isValid());
    assertTrue(Double.isFinite(view.toScreenX(5)));
    assertTrue(Double.isFinite(view.toScreenY(5)));
  }
}
