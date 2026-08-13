package tagbio.umap.fx;

import javafx.scene.paint.Color;

/**
 * Point colours for categorical labels.
 */
final class Palette {

  /** Chosen to stay distinguishable from each other against a light background. */
  private static final Color[] BASE = {
    Color.web("#0072b2"),
    Color.web("#d55e00"),
    Color.web("#009e73"),
    Color.web("#cc79a7"),
    Color.web("#e69f00"),
    Color.web("#56b4e9"),
    Color.web("#8c6d31"),
    Color.web("#7f3c8d"),
    Color.web("#b22222"),
    Color.web("#17a2b8"),
    Color.web("#6a8e1f"),
    Color.web("#4c4c4c"),
  };

  /** Golden angle in degrees, used to spread hues beyond the fixed palette. */
  private static final double GOLDEN_ANGLE = 137.508;

  private Palette() { }

  /**
   * The colour for a class.
   * @param index zero based class index
   * @return the colour to draw that class in
   */
  static Color forClass(final int index) {
    if (index < BASE.length) {
      return BASE[index];
    }
    // Stepping the hue by the golden angle keeps consecutive classes far apart in colour
    // instead of producing a slow gradient.
    return Color.hsb((index * GOLDEN_ANGLE) % 360, 0.62, 0.78);
  }
}
