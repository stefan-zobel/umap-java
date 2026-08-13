package tagbio.umap.fx;

/**
 * The mapping from embedding coordinates to screen pixels, including zoom and pan.
 *
 * <p>Deliberately free of JavaFX types: this is the only real arithmetic in the viewer, so
 * it is kept where it can be tested without starting a toolkit.
 */
final class ViewTransform {

  /** How far out of the fitted view zooming is allowed to go. */
  private static final double MIN_ZOOM = 0.25;
  /** How far into the fitted view zooming is allowed to go. */
  private static final double MAX_ZOOM = 400;
  private static final float MIN_RANGE = 1.0e-6f;

  private double mScale;
  private double mOffsetX;
  private double mOffsetY;
  /** Scale at which the whole embedding fits the viewport, used to bound zooming. */
  private double mFitScale;

  /**
   * Scale and centre so that the given data bounds are wholly visible.
   *
   * <p>Both axes get the same scale on purpose. Fitting x and y independently to the
   * viewport would distort the inter-cluster distances that make a UMAP projection
   * readable, so the shorter axis is centred within the leftover space instead.
   *
   * @param minX smallest x in the data
   * @param maxX largest x in the data
   * @param minY smallest y in the data
   * @param maxY largest y in the data
   * @param width viewport width in pixels
   * @param height viewport height in pixels
   * @param padding pixels to keep free on each edge
   */
  void fit(final float minX, final float maxX, final float minY, final float maxY,
    final double width, final double height, final double padding) {
    if (width <= 0 || height <= 0) {
      return;
    }
    final float rangeX = Math.max(maxX - minX, MIN_RANGE);
    final float rangeY = Math.max(maxY - minY, MIN_RANGE);
    mFitScale = Math.min(Math.max(width - 2 * padding, 1) / rangeX,
      Math.max(height - 2 * padding, 1) / rangeY);
    mScale = mFitScale;
    mOffsetX = (width - rangeX * mScale) / 2 - minX * mScale;
    mOffsetY = height - (height - rangeY * mScale) / 2 + minY * mScale;
  }

  /**
   * Whether a viewport has been fitted and drawing can proceed.
   * @return true once {@link #fit} has produced a usable scale
   */
  boolean isValid() {
    return mScale > 0;
  }

  /**
   * Horizontal screen position of an embedding coordinate.
   * @param x embedding x
   * @return pixel x
   */
  double toScreenX(final float x) {
    return mOffsetX + x * mScale;
  }

  /**
   * Vertical screen position of an embedding coordinate. Screen y grows downwards, so this
   * axis is flipped relative to the data.
   * @param y embedding y
   * @return pixel y
   */
  double toScreenY(final float y) {
    return mOffsetY - y * mScale;
  }

  /**
   * Shift the view by a pixel offset.
   * @param dx horizontal pixels
   * @param dy vertical pixels
   */
  void pan(final double dx, final double dy) {
    mOffsetX += dx;
    mOffsetY += dy;
  }

  /**
   * Change the scale by a factor while keeping whatever sits at the given pixel pinned
   * there, so that zooming follows the cursor instead of the centre.
   * @param cursorX pixel x to keep fixed
   * @param cursorY pixel y to keep fixed
   * @param factor multiplier for the current scale
   */
  void zoomAt(final double cursorX, final double cursorY, final double factor) {
    if (mFitScale <= 0) {
      return;
    }
    final double target = Math.max(mFitScale * MIN_ZOOM,
      Math.min(mFitScale * MAX_ZOOM, mScale * factor));
    final double applied = target / mScale;
    mOffsetX = cursorX - (cursorX - mOffsetX) * applied;
    mOffsetY = cursorY - (cursorY - mOffsetY) * applied;
    mScale = target;
  }
}
