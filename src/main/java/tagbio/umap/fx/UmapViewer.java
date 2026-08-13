package tagbio.umap.fx;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Callable;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.Slider;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import tagbio.umap.ProgressListener;
import tagbio.umap.ProgressState;
import tagbio.umap.Umap;
import tagbio.umap.UmapProgress;

/**
 * Computes a two dimensional UMAP projection of a delimited data file and shows it as
 * coloured points, one colour per class taken from the sample name prefix. The three
 * parameters that shape the layout most can be changed and the projection recomputed.
 *
 * <p>Launch through {@link Launcher} rather than calling this class directly, so that the
 * viewer also starts when JavaFX is on the classpath instead of the module path.
 */
public final class UmapViewer extends Application {

  private static final int NEAREST_NEIGHBOURS = 15;
  private static final float MIN_DIST = 0.1f;
  private static final float SPREAD = 1.0f;
  private static final long SEED = 42L;
  /**
   * Neighbour counts above this are not offered. Umap itself reduces the value to
   * <code>rows - 1</code> for small inputs, so no clamping to the data size is needed.
   */
  private static final double MAX_NEIGHBOURS = 100;

  private final EmbeddingCanvas mCanvas = new EmbeddingCanvas();
  private final Legend mLegend = new Legend();
  private final ProgressBar mProgressBar = new ProgressBar(0);
  private final Label mStatus = new Label();
  private final Slider mNeighbourSlider = new Slider(2, MAX_NEIGHBOURS, NEAREST_NEIGHBOURS);
  private final Slider mMinDistSlider = new Slider(0, ParameterRange.maxMinDist(SPREAD), MIN_DIST);
  private final Slider mSpreadSlider =
    new Slider(ParameterRange.MIN_SPREAD, ParameterRange.MAX_SPREAD, SPREAD);
  private final Label mNeighbourValue = new Label();
  private final Label mMinDistValue = new Label();
  private final Label mSpreadValue = new Label();

  private LabelledData mData;
  private boolean mFitting;
  /** Set while sliders are being adjusted programmatically, to suppress spurious refits. */
  private boolean mAdjusting;

  @Override
  public void start(final Stage stage) {
    stage.setTitle("UMAP projection");
    stage.setScene(new Scene(buildRoot(), 1180, 780));
    stage.show();

    final String file = dataFile(getParameters().getRaw());
    if (file == null) {
      mStatus.setText("Pass the path to a tab or comma separated data file as the first argument.");
      return;
    }
    loadAndFit(file);
  }

  private BorderPane buildRoot() {
    configureSlider(mNeighbourSlider, 1);
    configureSlider(mMinDistSlider, ParameterRange.STEP);
    configureSlider(mSpreadSlider, ParameterRange.STEP);
    updateNeighbourLabel();
    updateMinDistLabel();
    updateSpreadLabel();
    wireRecompute(mNeighbourSlider, this::updateNeighbourLabel);
    wireRecompute(mMinDistSlider, this::updateMinDistLabel);
    wireRecompute(mSpreadSlider, this::spreadChanged);

    mLegend.setOnVisibilityChanged(() -> mCanvas.setVisibleClasses(mLegend.getVisible()));

    final Button resetView = new Button("Reset view");
    resetView.setOnAction(event -> mCanvas.resetView());
    final Button exportPng = new Button("Export PNG...");
    exportPng.setOnAction(event -> exportPng());

    final HBox controls = new HBox(10,
      new Label("Neighbours"), mNeighbourSlider, mNeighbourValue,
      new Separator(Orientation.VERTICAL),
      new Label("Min dist"), mMinDistSlider, mMinDistValue,
      new Separator(Orientation.VERTICAL),
      new Label("Spread"), mSpreadSlider, mSpreadValue,
      new Separator(Orientation.VERTICAL),
      resetView, exportPng);
    controls.setAlignment(Pos.CENTER_LEFT);
    controls.setPadding(new Insets(8, 12, 8, 12));

    final ScrollPane legendPane = new ScrollPane(mLegend);
    legendPane.setFitToWidth(true);
    legendPane.setMinWidth(170);

    final Label hint =
      new Label("scroll to zoom · drag to pan · hover a point for details · click a class to hide it");
    hint.setStyle("-fx-text-fill: #707070;");
    final HBox statusBar = new HBox(10, mProgressBar, mStatus, new Separator(Orientation.VERTICAL), hint);
    statusBar.setAlignment(Pos.CENTER_LEFT);
    statusBar.setPadding(new Insets(6, 12, 6, 12));

    final BorderPane root = new BorderPane();
    root.setTop(controls);
    root.setCenter(mCanvas);
    root.setRight(legendPane);
    root.setBottom(statusBar);
    return root;
  }

  private static void configureSlider(final Slider slider, final double tickUnit) {
    slider.setPrefWidth(170);
    slider.setMajorTickUnit(tickUnit);
    slider.setMinorTickCount(0);
    slider.setBlockIncrement(tickUnit);
    slider.setSnapToTicks(true);
  }

  /**
   * Recompute when the user lets go of a slider, rather than on every intermediate value,
   * so that dragging does not queue up a fit per pixel.
   */
  private void wireRecompute(final Slider slider, final Runnable labelUpdate) {
    slider.valueProperty().addListener((observable, oldValue, newValue) -> {
      labelUpdate.run();
      // A click on the track or an arrow key changes the value without a drag.
      if (!slider.isValueChanging()) {
        refit();
      }
    });
    slider.valueChangingProperty().addListener((observable, wasChanging, isChanging) -> {
      if (!isChanging) {
        refit();
      }
    });
  }

  /**
   * The usable range of the minimum distance depends on the spread, so the one slider has
   * to follow the other. The adjustment is marked so that clamping the minimum distance
   * does not itself trigger a projection.
   */
  private void spreadChanged() {
    updateSpreadLabel();
    mAdjusting = true;
    try {
      mMinDistSlider.setMax(ParameterRange.maxMinDist(spread()));
    } finally {
      mAdjusting = false;
    }
    updateMinDistLabel();
  }

  private void updateNeighbourLabel() {
    mNeighbourValue.setText(String.valueOf(neighbours()));
  }

  private void updateMinDistLabel() {
    mMinDistValue.setText(String.format("%.2f", minDist()));
  }

  private void updateSpreadLabel() {
    mSpreadValue.setText(String.format("%.2f", spread()));
  }

  private int neighbours() {
    return (int) Math.round(mNeighbourSlider.getValue());
  }

  private float minDist() {
    return (float) mMinDistSlider.getValue();
  }

  private float spread() {
    return (float) mSpreadSlider.getValue();
  }

  private static String dataFile(final List<String> args) {
    for (final String arg : args) {
      if (!arg.trim().isEmpty()) {
        return arg.trim();
      }
    }
    return null;
  }

  private void loadAndFit(final String file) {
    final Settings settings = currentSettings();
    startFit(() -> {
      final LabelledData data = LabelledData.load(file);
      return settings.project(data);
    });
  }

  private void refit() {
    final LabelledData data = mData;
    if (data == null || mFitting || mAdjusting) {
      return;
    }
    final Settings settings = currentSettings();
    startFit(() -> settings.project(data));
  }

  /**
   * Read the slider values on the application thread, where JavaFX properties belong, and
   * hand the background thread a fixed set of parameters.
   */
  private Settings currentSettings() {
    return new Settings(neighbours(), minDist(), spread());
  }

  private void startFit(final Callable<Projection> work) {
    mFitting = true;
    setControlsDisabled(true);
    mProgressBar.setProgress(0);
    mStatus.setText("Fitting...");

    final Task<Projection> task = new Task<Projection>() {
      @Override
      protected Projection call() throws Exception {
        return work.call();
      }
    };

    // UmapProgress is a global singleton, so the listener has to come off again once the
    // fit has finished.
    final ProgressListener listener = (final ProgressState state) -> {
      final int total = state.getTotal();
      final double fraction = total > 0 ? state.getCount() / (double) total : -1;
      Platform.runLater(() -> mProgressBar.setProgress(fraction));
    };
    UmapProgress.addProgressListener(listener);

    task.setOnSucceeded(event -> {
      UmapProgress.removeProgressListener(listener);
      final Projection projection = task.getValue();
      mData = projection.getData();
      mCanvas.setProjection(projection);
      mLegend.setClasses(mData.getClassNames());
      mProgressBar.setProgress(1);
      // Reported from the projection, not from the sliders: the two can differ if the
      // sliders moved again while this fit was running.
      mStatus.setText(String.format("%d points, %d classes, neighbours %d, min dist %.2f, spread %.2f",
        projection.getEmbedding().length, mData.getClassNames().length,
        projection.getNeighbours(), projection.getMinDist(), projection.getSpread()));
      finishFit();
    });
    task.setOnFailed(event -> {
      UmapProgress.removeProgressListener(listener);
      mProgressBar.setProgress(0);
      final Throwable error = task.getException();
      mStatus.setText("Failed: " + (error == null ? "unknown error" : error.toString()));
      finishFit();
    });

    // Fitting must not run on the application thread or the window stays frozen for the
    // whole computation.
    final Thread thread = new Thread(task, "umap-fit");
    thread.setDaemon(true);
    thread.start();
  }

  /**
   * Write the plot to a PNG, as it currently looks. The exported resolution is the size of
   * the plot area on screen, so a larger window yields a larger image.
   */
  private void exportPng() {
    if (!mCanvas.hasProjection()) {
      mStatus.setText("Nothing to export yet.");
      return;
    }
    final FileChooser chooser = new FileChooser();
    chooser.setTitle("Export PNG");
    chooser.setInitialFileName("umap.png");
    chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("PNG image", "*.png"));
    final File file = chooser.showSaveDialog(mCanvas.getScene().getWindow());
    if (file == null) {
      return;
    }
    try {
      ImageExport.writePng(mCanvas.snapshotPlot(), file);
      mStatus.setText("Wrote " + file.getAbsolutePath());
    } catch (final IOException e) {
      mStatus.setText("Export failed: " + e);
    }
  }

  private void finishFit() {
    mFitting = false;
    setControlsDisabled(false);
  }

  private void setControlsDisabled(final boolean disabled) {
    mNeighbourSlider.setDisable(disabled);
    mMinDistSlider.setDisable(disabled);
    mSpreadSlider.setDisable(disabled);
  }

  /** One immutable set of projection parameters, safe to hand to a background thread. */
  private static final class Settings {
    private final int mNeighbours;
    private final float mMinDist;
    private final float mSpread;

    private Settings(final int neighbours, final float minDist, final float spread) {
      mNeighbours = neighbours;
      mMinDist = minDist;
      mSpread = spread;
    }

    private Projection project(final LabelledData data) {
      // A fixed seed on a single thread keeps the picture reproducible; more than one
      // thread makes the embedding differ between runs.
      final Umap umap = new Umap()
        .setNumberComponents(2)
        .setNumberNearestNeighbours(mNeighbours)
        .setMinDist(mMinDist)
        .setSpread(mSpread)
        .setSeed(SEED)
        .setThreads(1);
      return new Projection(data, umap.fitTransform(data.getData()), mNeighbours, mMinDist, mSpread);
    }
  }
}
