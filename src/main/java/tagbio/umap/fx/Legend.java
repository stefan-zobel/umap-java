package tagbio.umap.fx;

import java.util.Arrays;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Cursor;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;

/**
 * Lists class names next to the colour their points are drawn in. Clicking a row hides or
 * shows that class in the plot.
 */
final class Legend extends VBox {

  private static final double SWATCH_SIZE = 11;
  private static final double HIDDEN_OPACITY = 0.4;

  private final Button mShowAll = new Button("Show all");

  private boolean[] mVisible = new boolean[0];
  private HBox[] mRows = new HBox[0];
  private Rectangle[] mSwatches = new Rectangle[0];
  private Runnable mListener;

  Legend() {
    setSpacing(4);
    setPadding(new Insets(12));
    mShowAll.setOnAction(event -> {
      Arrays.fill(mVisible, true);
      refresh();
      notifyListener();
    });
  }

  /**
   * Register a callback invoked whenever the set of visible classes changes.
   * @param listener called after {@link #getVisible} has been updated
   */
  void setOnVisibilityChanged(final Runnable listener) {
    mListener = listener;
  }

  /**
   * Which classes are currently shown, indexed by class.
   * @return one flag per class
   */
  boolean[] getVisible() {
    return mVisible;
  }

  /**
   * Replace the listed classes. All of them start out visible.
   * @param classNames class names, in the same order as the class indices used for drawing
   */
  void setClasses(final String[] classNames) {
    mVisible = new boolean[classNames.length];
    Arrays.fill(mVisible, true);
    mRows = new HBox[classNames.length];
    mSwatches = new Rectangle[classNames.length];

    getChildren().clear();
    final Label title = new Label("Classes");
    title.setStyle("-fx-font-weight: bold;");
    getChildren().add(title);

    for (int i = 0; i < classNames.length; ++i) {
      final int index = i;
      final Rectangle swatch = new Rectangle(SWATCH_SIZE, SWATCH_SIZE, Palette.forClass(i));
      swatch.setStrokeWidth(1.5);
      final HBox row = new HBox(6, swatch, new Label(classNames[i]));
      row.setAlignment(Pos.CENTER_LEFT);
      row.setCursor(Cursor.HAND);
      row.setOnMouseClicked(event -> toggle(index));
      mSwatches[i] = swatch;
      mRows[i] = row;
      getChildren().add(row);
    }

    if (classNames.length > 1) {
      final VBox spacer = new VBox();
      spacer.setPadding(new Insets(6, 0, 0, 0));
      getChildren().addAll(spacer, mShowAll);
    }
    refresh();
  }

  private void toggle(final int index) {
    mVisible[index] = !mVisible[index];
    // Never let the last visible class be switched off: an empty plot looks like a bug.
    if (!anyVisible()) {
      mVisible[index] = true;
      return;
    }
    refresh();
    notifyListener();
  }

  private boolean anyVisible() {
    for (final boolean visible : mVisible) {
      if (visible) {
        return true;
      }
    }
    return false;
  }

  /** Hidden classes keep their place in the list but are dimmed and drawn as an outline. */
  private void refresh() {
    for (int i = 0; i < mRows.length; ++i) {
      final boolean visible = mVisible[i];
      mRows[i].setOpacity(visible ? 1.0 : HIDDEN_OPACITY);
      mSwatches[i].setFill(visible ? Palette.forClass(i) : Color.TRANSPARENT);
      mSwatches[i].setStroke(visible ? Color.TRANSPARENT : Palette.forClass(i));
    }
    mShowAll.setDisable(noneHidden());
  }

  private boolean noneHidden() {
    for (final boolean visible : mVisible) {
      if (!visible) {
        return false;
      }
    }
    return true;
  }

  private void notifyListener() {
    if (mListener != null) {
      mListener.run();
    }
  }
}
