package tagbio.umap.fx;

import javafx.scene.Cursor;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;

/**
 * Draws a two dimensional embedding as coloured points, with zoom, pan and a hover readout
 * of the point under the cursor.
 *
 * <p>A canvas is used rather than a chart node because a scatter chart allocates a scene
 * graph node per point, which does not scale to the row counts UMAP is normally used on.
 * The coordinate arithmetic lives in {@link ViewTransform}.
 */
final class EmbeddingCanvas extends Region {

  private static final Color BACKGROUND = Color.web("#fdfdfd");
  private static final double PADDING = 24;
  private static final double POINT_RADIUS = 2.6;
  private static final double POINT_ALPHA = 0.75;
  private static final double ZOOM_STEP = 1.15;
  /** How close the cursor has to be, in pixels, for a point to be picked. */
  private static final double PICK_RADIUS = 10;
  private static final double TOOLTIP_GAP = 14;
  private static final String TOOLTIP_STYLE =
    "-fx-background-color: rgba(255, 255, 255, 0.94);"
      + "-fx-border-color: #b0b0b0;"
      + "-fx-border-radius: 3;"
      + "-fx-background-radius: 3;"
      + "-fx-padding: 5 8 5 8;"
      + "-fx-font-size: 11;";

  private final Canvas mCanvas = new Canvas();
  private final Label mReadout = new Label();
  private final ViewTransform mView = new ViewTransform();

  private Projection mProjection;
  private float[][] mPoints;
  private int[] mClassIndices;
  /**
   * The point indices of each class. The draw loop runs one class at a time, and grouping them
   * up front turns that from a full scan per class into a single pass. At the ten classes and
   * 60 000 points of MNIST that is the difference between 600 000 and 60 000 iterations for
   * every pan, zoom and resize.
   */
  private int[][] mByClass;
  /** Null means every class is shown. */
  private boolean[] mVisibleClasses;

  private float mMinX;
  private float mMaxX;
  private float mMinY;
  private float mMaxY;

  private boolean mUserAdjusted;
  private double mDragX;
  private double mDragY;

  EmbeddingCanvas() {
    mCanvas.setManaged(false);
    mReadout.setManaged(false);
    mReadout.setMouseTransparent(true);
    mReadout.setVisible(false);
    mReadout.setStyle(TOOLTIP_STYLE);
    getChildren().addAll(mCanvas, mReadout);
    setMinSize(240, 240);
    setPrefSize(820, 640);

    setOnScroll(event -> {
      if (event.getDeltaY() != 0) {
        mView.zoomAt(event.getX(), event.getY(), event.getDeltaY() > 0 ? ZOOM_STEP : 1 / ZOOM_STEP);
        mUserAdjusted = true;
        redraw();
        updateReadout(event.getX(), event.getY());
      }
    });
    setOnMousePressed(event -> {
      mDragX = event.getX();
      mDragY = event.getY();
      setCursor(Cursor.MOVE);
      hideReadout();
    });
    setOnMouseReleased(event -> setCursor(Cursor.DEFAULT));
    setOnMouseDragged(event -> {
      mView.pan(event.getX() - mDragX, event.getY() - mDragY);
      mDragX = event.getX();
      mDragY = event.getY();
      mUserAdjusted = true;
      redraw();
    });
    setOnMouseMoved(event -> updateReadout(event.getX(), event.getY()));
    setOnMouseExited(event -> hideReadout());
  }

  /**
   * Show an embedding, replacing anything drawn before and resetting the view.
   * @param projection the embedding and its labels
   */
  void setProjection(final Projection projection) {
    mProjection = projection;
    mPoints = projection.getEmbedding();
    mClassIndices = projection.getData().getClassIndices();
    // The class count can change with the data, so any previous filter is dropped.
    mVisibleClasses = null;
    hideReadout();

    groupByClass(projection.getData().getClassNames().length);

    mMinX = 0;
    mMaxX = 0;
    mMinY = 0;
    mMaxY = 0;
    if (mPoints.length > 0 && mPoints[0].length >= 2) {
      mMinX = mPoints[0][0];
      mMaxX = mMinX;
      mMinY = mPoints[0][1];
      mMaxY = mMinY;
      for (final float[] point : mPoints) {
        mMinX = Math.min(mMinX, point[0]);
        mMaxX = Math.max(mMaxX, point[0]);
        mMinY = Math.min(mMinY, point[1]);
        mMaxY = Math.max(mMaxY, point[1]);
      }
    }
    resetView();
  }

  /** Bucket the point indices by class in one pass, counting first so nothing has to grow. */
  private void groupByClass(final int classCount) {
    final int[] counts = new int[classCount];
    for (final int cls : mClassIndices) {
      ++counts[cls];
    }
    mByClass = new int[classCount][];
    for (int cls = 0; cls < classCount; ++cls) {
      mByClass[cls] = new int[counts[cls]];
    }
    final int[] filled = new int[classCount];
    for (int i = 0; i < mClassIndices.length; ++i) {
      final int cls = mClassIndices[i];
      mByClass[cls][filled[cls]++] = i;
    }
  }

  /**
   * Restrict drawing and picking to a subset of the classes.
   * @param visible one flag per class, or null to show all of them
   */
  void setVisibleClasses(final boolean[] visible) {
    mVisibleClasses = visible == null ? null : visible.clone();
    hideReadout();
    redraw();
  }

  /**
   * Whether an embedding has been set and there is something to draw or export.
   * @return true once a projection has arrived
   */
  boolean hasProjection() {
    return mProjection != null;
  }

  /**
   * Render the plot exactly as displayed, at the current size, zoom and class filter. The
   * hover readout is a sibling of the canvas rather than part of it, so it is never
   * captured.
   * @return the rendered plot
   */
  Image snapshotPlot() {
    return mCanvas.snapshot(null, null);
  }

  private boolean isClassVisible(final int cls) {
    return mVisibleClasses == null || cls >= mVisibleClasses.length || mVisibleClasses[cls];
  }

  /**
   * Scale and centre the embedding so that all of it is visible again.
   */
  void resetView() {
    mUserAdjusted = false;
    fitView();
    redraw();
  }

  private void fitView() {
    if (mPoints != null) {
      mView.fit(mMinX, mMaxX, mMinY, mMaxY, mCanvas.getWidth(), mCanvas.getHeight(), PADDING);
    }
  }

  @Override
  protected void layoutChildren() {
    mCanvas.relocate(0, 0);
    mCanvas.setWidth(getWidth());
    mCanvas.setHeight(getHeight());
    // Refit on resize, unless the user has zoomed or panned and would lose their place.
    if (!mUserAdjusted) {
      fitView();
    }
    redraw();
  }

  private void redraw() {
    final GraphicsContext gfx = mCanvas.getGraphicsContext2D();
    final double width = mCanvas.getWidth();
    final double height = mCanvas.getHeight();
    gfx.setGlobalAlpha(1.0);
    gfx.setFill(BACKGROUND);
    gfx.fillRect(0, 0, width, height);
    if (mPoints == null || mPoints.length == 0 || mPoints[0].length < 2 || !mView.isValid()) {
      return;
    }

    gfx.setGlobalAlpha(POINT_ALPHA);
    // Point size stays constant in screen pixels so that zooming in resolves dense clusters
    // instead of magnifying blobs.
    final double diameter = 2 * POINT_RADIUS;
    // Outer loop over classes so the fill colour is set once per class rather than once per
    // point, and so hiding a class costs one check instead of one per point.
    for (int cls = 0; cls < mByClass.length; ++cls) {
      if (!isClassVisible(cls)) {
        continue;
      }
      gfx.setFill(Palette.forClass(cls));
      for (final int i : mByClass[cls]) {
        final double px = mView.toScreenX(mPoints[i][0]);
        final double py = mView.toScreenY(mPoints[i][1]);
        if (px >= -POINT_RADIUS && px <= width + POINT_RADIUS
          && py >= -POINT_RADIUS && py <= height + POINT_RADIUS) {
          gfx.fillOval(px - POINT_RADIUS, py - POINT_RADIUS, diameter, diameter);
        }
      }
    }
    gfx.setGlobalAlpha(1.0);
  }

  /**
   * The index of the point nearest the cursor, searched in screen space so that the pick
   * radius stays a constant number of pixels at every zoom level. A linear scan is enough:
   * it costs a subtraction and a multiply per point and runs only on mouse movement.
   */
  private int pickNearest(final double cursorX, final double cursorY) {
    if (mPoints == null || !mView.isValid()) {
      return -1;
    }
    double best = PICK_RADIUS * PICK_RADIUS;
    int bestIndex = -1;
    for (int i = 0; i < mPoints.length; ++i) {
      // A hidden point must not be pickable, or the readout describes something invisible.
      if (!isClassVisible(mClassIndices[i])) {
        continue;
      }
      final double dx = mView.toScreenX(mPoints[i][0]) - cursorX;
      final double dy = mView.toScreenY(mPoints[i][1]) - cursorY;
      final double distance = dx * dx + dy * dy;
      if (distance < best) {
        best = distance;
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  private void updateReadout(final double cursorX, final double cursorY) {
    final int index = pickNearest(cursorX, cursorY);
    if (index < 0) {
      hideReadout();
      return;
    }
    final PointData data = mProjection.getData();
    final String className = data.getClassNames()[data.getClassIndices()[index]];
    mReadout.setText(data.getSampleNames()[index] + '\n'
      + "class: " + className + '\n'
      + String.format("x %.3f  y %.3f", mPoints[index][0], mPoints[index][1]));
    mReadout.setVisible(true);
    // Without applying the style first the preferred size is still unknown on the very
    // first hover, and the clamping below would use stale measurements.
    mReadout.applyCss();
    mReadout.autosize();

    // Keep the readout inside the viewport rather than letting it run off the edge.
    final double placedX = Math.min(cursorX + TOOLTIP_GAP, getWidth() - mReadout.getWidth() - 2);
    final double placedY = Math.min(cursorY + TOOLTIP_GAP, getHeight() - mReadout.getHeight() - 2);
    mReadout.relocate(Math.max(2, placedX), Math.max(2, placedY));
  }

  private void hideReadout() {
    mReadout.setVisible(false);
  }
}
