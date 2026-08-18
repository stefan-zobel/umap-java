package tagbio.umap.fx;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import javafx.animation.Animation;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Separator;
import javafx.scene.control.Slider;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.util.Duration;

import tagbio.umap.ProgressListener;
import tagbio.umap.ProgressState;
import tagbio.umap.Umap;
import tagbio.umap.UmapProgress;

/**
 * Computes a two dimensional UMAP projection and shows it as coloured points, one colour per
 * class. The three parameters that shape the layout most can be changed and the projection
 * recomputed.
 *
 * <p>The points come from either a delimited data file, where the class is the sample name
 * prefix, or from the MNIST IDX pair, where it is the digit. Which one is decided from the
 * launch argument.
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

  /**
   * Fewest MNIST images offered, and the step the slider moves in.
   *
   * <p>Not an arbitrary floor. Umap computes the full pairwise distance matrix below
   * <code>SMALL_PROBLEM_THRESHOLD</code>, which is 4096 rows, and the approximate neighbour
   * descent above it. Measured on this data, 4000 points take 6.9 s and 5000 take 5.0 s - more
   * points, less time - and the matrix alone is 67 MB just under the threshold. Starting at
   * 5000 keeps the whole offered range on the side where the viewer gets steadily slower as it
   * is given more to do, instead of having its worst case in the middle.
   */
  private static final int MIN_POINTS = 5000;

  /** Images loaded unless the slider is moved. 10 000 of them project in about 10 s. */
  private static final int DEFAULT_POINTS = 10000;

  /** System property carrying the requested thread count; the POM passes it through. */
  static final String THREADS_PROPERTY = "umap.threads";

  /**
   * Logical cores, read once. It is both the largest thread count the dropdown offers and the
   * cap on what the POM may ask for, so there is no setting under which the viewer can ask for
   * more threads than the machine has.
   */
  private static final int CORES = Math.max(1, Runtime.getRuntime().availableProcessors());

  private final EmbeddingCanvas mCanvas = new EmbeddingCanvas();
  private final Legend mLegend = new Legend();
  private final ProgressBar mProgressBar = new ProgressBar(0);
  private final Label mStatus = new Label();
  private final Slider mNeighbourSlider = new Slider(2, MAX_NEIGHBOURS, NEAREST_NEIGHBOURS);
  private final Slider mMinDistSlider = new Slider(0, ParameterRange.maxMinDist(SPREAD), MIN_DIST);
  private final Slider mSpreadSlider =
    new Slider(ParameterRange.MIN_SPREAD, ParameterRange.MAX_SPREAD, SPREAD);
  private final Slider mPointsSlider = new Slider(MIN_POINTS, DEFAULT_POINTS, DEFAULT_POINTS);
  private final Label mNeighbourValue = new Label();
  private final Label mMinDistValue = new Label();
  private final Label mSpreadValue = new Label();
  private final Label mPointsValue = new Label();

  /**
   * Threads the next projection may use, one to the logical core count. What it starts on comes
   * from {@link #resolveThreads}; from then on it is the user's to change, and changing it
   * recomputes like any other control. Anything above one trades the reproducible embedding
   * away, which is why the count is on screen next to the picture rather than only in the POM.
   */
  private final ComboBox<Integer> mThreadChoice = new ComboBox<>();

  /**
   * Runs the same settings again. Every other control recomputes as a side effect of changing
   * something, so without this there is no way to ask for a second run of one configuration -
   * and that is the only way to see what the thread count costs. The two pictures are drawn in
   * the same frame, because a new projection resets the view.
   *
   * <p>Measured on the 1797 digits at the viewer's own settings, twice at each count: at one
   * thread 0 of 3594 coordinates differ, and at six all 3594 do, by 3.6 on average against a
   * map about 13 across. So the difference the button shows is a quarter of the plot, not a
   * last-bit wobble - which is what makes it worth a button rather than a footnote.
   */
  private final Button mRecompute = new Button("Recompute");

  /** Set when the argument names a directory holding the MNIST IDX pair. */
  private File mMnistDirectory;
  /** Set when the argument names a delimited data file. */
  private String mDataFile;

  /** False when the source fixes the point count, so that no fit may re-enable the slider. */
  private boolean mPointsAdjustable;

  /**
   * Ticks once a second while a fit is running, so that the status line keeps moving even
   * through a phase that reports nothing.
   *
   * <p>A bar alone is not enough here. Umap builds its progress total in three goes - it
   * resets to 5, adds the trees and the descent iterations, and only then adds the epochs
   * (<code>Umap.java:1103</code>, <code>:232</code>, <code>:1191</code>) - so the fraction
   * jumps backwards twice per run as the denominator grows. That is over in a blink on a small
   * file and is on screen for minutes at sixty thousand points, where a bar pinned near the
   * left is indistinguishable from a hang.
   */
  private final Timeline mElapsed =
    new Timeline(new KeyFrame(Duration.seconds(1), event -> showElapsed()));
  private long mFitStart;
  private String mFitDescription = "";
  /** Latest progress, written by the listener and read by the ticker. */
  private volatile int mProgressCount;
  private volatile int mProgressTotal;

  private boolean mFitting;
  /** Set while sliders are being adjusted programmatically, to suppress spurious refits. */
  private boolean mAdjusting;

  @Override
  public void start(final Stage stage) {
    stage.setTitle("UMAP projection");
    stage.setScene(new Scene(buildRoot(), 1180, 820));
    stage.show();

    final String argument = dataArgument(getParameters().getRaw());
    if (argument == null) {
      mStatus.setText("Pass a data file, or a directory holding the MNIST IDX files, as the first argument.");
      return;
    }
    final File path = new File(argument);
    if (MnistData.isMnistDirectory(path)) {
      if (!configureForMnist(path)) {
        return;
      }
    } else {
      mDataFile = argument;
      // The whole file is projected, so there is nothing for the point count to select.
      mPointsSlider.setDisable(true);
      mPointsValue.setText("file");
    }
    refit();
  }

  /**
   * Point the viewer at MNIST and size the point slider to what the file actually holds.
   * @param directory the directory holding the two IDX files
   * @return false if the images could not be read, in which case the status line says so
   */
  private boolean configureForMnist(final File directory) {
    final int available;
    try {
      // Only the header is read, so this is cheap enough to do on the application thread.
      available = MnistData.available(directory);
    } catch (final IOException e) {
      mStatus.setText("Cannot read the MNIST images: " + e.getMessage());
      return false;
    }
    mMnistDirectory = directory;
    mAdjusting = true;
    try {
      if (available <= MIN_POINTS) {
        // Fewer images than the slider's own floor: there is nothing to choose between, so it
        // is pinned to what the file holds. Collapsing the range rather than lowering only the
        // maximum matters, because a minimum left above the value would clamp it back up and
        // the loader would then ask for more records than there are.
        mPointsSlider.setMin(available);
        mPointsSlider.setMax(available);
        mPointsSlider.setValue(available);
        mPointsSlider.setDisable(true);
      } else {
        mPointsAdjustable = true;
        mPointsSlider.setMax(available);
        mPointsSlider.setValue(Math.min(DEFAULT_POINTS, available));
      }
    } finally {
      mAdjusting = false;
    }
    updatePointsLabel();
    return true;
  }

  private BorderPane buildRoot() {
    mElapsed.setCycleCount(Animation.INDEFINITE);
    configureSlider(mNeighbourSlider, 1);
    configureSlider(mMinDistSlider, ParameterRange.STEP);
    configureSlider(mSpreadSlider, ParameterRange.STEP);
    configureSlider(mPointsSlider, MIN_POINTS);
    updateNeighbourLabel();
    updateMinDistLabel();
    updateSpreadLabel();
    updatePointsLabel();
    wireRecompute(mNeighbourSlider, this::updateNeighbourLabel);
    wireRecompute(mMinDistSlider, this::updateMinDistLabel);
    wireRecompute(mSpreadSlider, this::spreadChanged);
    wireRecompute(mPointsSlider, this::updatePointsLabel);
    configureThreadChoice();

    mLegend.setOnVisibilityChanged(() -> mCanvas.setVisibleClasses(mLegend.getVisible()));

    mRecompute.setOnAction(event -> refit());
    // Dead until there is something to recompute. The sliders can afford to be live with no
    // data because moving one is not an instruction to do anything; pressing a button is, and
    // a button that answers nothing reads as a broken one. The first fit to finish - or fail -
    // enables it through setControlsDisabled.
    mRecompute.setDisable(true);
    mRecompute.setTooltip(new Tooltip(
      "Project the same data with the same settings again.\n"
        + "At one thread the picture is identical; above one it is not."));
    final Button resetView = new Button("Reset view");
    resetView.setOnAction(event -> mCanvas.resetView());
    final Button exportPng = new Button("Export PNG...");
    exportPng.setOnAction(event -> exportPng());

    final HBox parameters = new HBox(10,
      new Label("Neighbours"), mNeighbourSlider, mNeighbourValue,
      new Separator(Orientation.VERTICAL),
      new Label("Min dist"), mMinDistSlider, mMinDistValue,
      new Separator(Orientation.VERTICAL),
      new Label("Spread"), mSpreadSlider, mSpreadValue);
    parameters.setAlignment(Pos.CENTER_LEFT);

    final HBox actions = new HBox(10,
      new Label("Points"), mPointsSlider, mPointsValue,
      new Separator(Orientation.VERTICAL),
      new Label("Threads"), mThreadChoice, mRecompute,
      new Separator(Orientation.VERTICAL),
      resetView, exportPng);
    actions.setAlignment(Pos.CENTER_LEFT);

    final VBox controls = new VBox(8, parameters, actions);
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

  /**
   * Fill the thread dropdown and select what the POM asked for.
   *
   * <p>The selection is made before the listener is attached, so that starting up does not
   * itself look like a change and queue a projection the viewer has no data for yet.
   */
  private void configureThreadChoice() {
    mThreadChoice.getItems().setAll(threadChoices(CORES));
    mThreadChoice.setValue(resolveThreads(requestedThreads(), CORES));
    mThreadChoice.valueProperty().addListener((observable, oldValue, newValue) -> refit());
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

  private void updatePointsLabel() {
    if (mDataFile == null) {
      mPointsValue.setText(String.valueOf(points()));
    }
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

  private int points() {
    return (int) Math.round(mPointsSlider.getValue());
  }

  /**
   * Threads the next projection will use, as chosen in the dropdown.
   *
   * @return the selected count, never below one
   */
  private int threads() {
    final Integer selected = mThreadChoice.getValue();
    return selected == null ? 1 : selected;
  }

  /**
   * The thread count asked for through {@value #THREADS_PROPERTY}, or null when the property
   * carries nothing usable - absent, empty, or not an integer. A malformed value is treated as
   * absent rather than fatal: it is a tuning knob, and refusing to start over it would be out
   * of proportion.
   *
   * @return the requested count, or null if nothing was requested
   */
  static Integer requestedThreads() {
    return Integer.getInteger(THREADS_PROPERTY);
  }

  /**
   * How many threads the viewer starts on. The dropdown can change it afterwards; this only
   * decides what it is set to before anyone touches it.
   *
   * <p>The property saying nothing and the property saying one are different requests, and are
   * answered differently. <b>A value in the POM</b> is deliberate and keeps the rule it always
   * had: <code>min(requested, cores)</code>, honoured up to what the machine actually has, and
   * at or below zero it is one - the only setting under which the embedding is reproducible.
   * <b>Nothing in the POM</b> is not a request for one thread, it is no request at all, and the
   * viewer then starts on <code>max(1, cores / 2)</code>: worth having over one thread, and it
   * leaves the machine usable while a fit that can run for a minute is running.
   *
   * <p>The core count is a parameter rather than read here so that the rule can be asserted
   * without depending on the machine running the test.
   *
   * @param requested the value of the system property, null if nothing was configured
   * @param cores the number of logical cores available
   * @return the thread count to start on, never below one and never above the core count
   */
  static int resolveThreads(final Integer requested, final int cores) {
    final int available = Math.max(cores, 1);
    if (requested == null) {
      return Math.max(1, available / 2);
    }
    if (requested <= 0) {
      return 1;
    }
    return Math.min(requested, available);
  }

  /**
   * The thread counts the dropdown offers, one to the logical core count.
   *
   * <p>Built here rather than in the caller so that the offered range can be asserted without a
   * JavaFX toolkit - and with it the one thing that would show as an empty box on screen, that
   * {@link #resolveThreads} always returns a value this list contains.
   *
   * @param cores the number of logical cores available
   * @return the counts to offer, always starting at one
   */
  static List<Integer> threadChoices(final int cores) {
    final List<Integer> choices = new ArrayList<>();
    for (int threads = 1; threads <= Math.max(cores, 1); ++threads) {
      choices.add(threads);
    }
    return choices;
  }

  private static String dataArgument(final List<String> args) {
    for (final String arg : args) {
      if (!arg.trim().isEmpty()) {
        return arg.trim();
      }
    }
    return null;
  }

  private void refit() {
    if (mFitting || mAdjusting || (mMnistDirectory == null && mDataFile == null)) {
      return;
    }
    final Settings settings = currentSettings();
    final Callable<PointData> loader = currentLoader();
    mFitDescription = describeRequest();
    startFit(() -> settings.project(loader.call()));
  }

  /**
   * What the run about to start was asked for, for the status line while it is running.
   *
   * <p>The thread count is named only when it is more than one, because that is when it is
   * worth knowing: it is the number the elapsed seconds next to it have to be read against.
   */
  private String describeRequest() {
    final StringBuilder text = new StringBuilder();
    if (mMnistDirectory != null) {
      text.append(points()).append(" points, ");
    }
    text.append("neighbours ").append(neighbours());
    if (threads() > 1) {
      text.append(", ").append(threads()).append(" threads");
    }
    return text.toString();
  }

  /**
   * Keep the status line moving whether or not the algorithm has anything to report.
   *
   * <p>Measured on 60 000 points at 50 neighbours, a 157 s run leaves the progress fraction on
   * one value for 42 s and on the next for another 23 s: the whole graph construction between
   * the neighbour search and the epochs reports twice. A bar alone therefore says "hung" for
   * two thirds of a minute at a stretch. The seconds here keep counting through that, and the
   * raw step is shown next to them so that the denominator growing mid-run is legible rather
   * than mysterious.
   */
  private void showElapsed() {
    final int total = mProgressTotal;
    final String step = total > 0 ? String.format("step %d/%d, ", mProgressCount, total) : "";
    mStatus.setText(String.format("Fitting %s - %s%d s",
      mFitDescription, step, (System.currentTimeMillis() - mFitStart) / 1000));
  }

  /**
   * Read the control values on the application thread, where JavaFX properties belong, and
   * hand the background thread a fixed set of parameters.
   */
  private Settings currentSettings() {
    return new Settings(neighbours(), minDist(), spread(), threads());
  }

  /**
   * How the next fit is to get its points, resolved on the application thread so that the
   * worker never reads a control.
   *
   * <p>The data is read again for every fit rather than kept. Re-reading is well under a
   * second either way, against a projection measured in seconds to a minute, and holding it
   * would mean deciding when a cached table has gone stale against the point count.
   *
   * @return a loader safe to call from a background thread
   */
  private Callable<PointData> currentLoader() {
    final File directory = mMnistDirectory;
    if (directory != null) {
      final int points = points();
      return () -> MnistData.load(directory, points);
    }
    final String file = mDataFile;
    return () -> LabelledData.load(file);
  }

  private void startFit(final Callable<Projection> work) {
    mFitting = true;
    setControlsDisabled(true);
    // Indeterminate, not zero. See the comment on mElapsed: the fraction is not monotonic, so
    // a determinate bar would visibly run backwards. The barber pole says only "still working",
    // which is the one thing it can say truthfully throughout.
    mProgressBar.setProgress(ProgressBar.INDETERMINATE_PROGRESS);
    mProgressCount = 0;
    mProgressTotal = 0;
    mFitStart = System.currentTimeMillis();
    showElapsed();
    mElapsed.playFromStart();

    final Task<Projection> task = new Task<Projection>() {
      @Override
      protected Projection call() throws Exception {
        return work.call();
      }
    };

    // UmapProgress is a global singleton, so the listener has to come off again once the
    // fit has finished.
    // Only records; every control is touched by the ticker on the application thread, so no
    // hop is needed here and a burst of notifications cannot flood the event queue.
    final ProgressListener listener = (final ProgressState state) -> {
      mProgressCount = state.getCount();
      mProgressTotal = state.getTotal();
    };
    UmapProgress.addProgressListener(listener);

    task.setOnSucceeded(event -> {
      UmapProgress.removeProgressListener(listener);
      final Projection projection = task.getValue();
      mCanvas.setProjection(projection);
      mLegend.setClasses(projection.getData().getClassNames());
      mProgressBar.setProgress(1);
      // Reported from the projection, not from the controls: the two can differ if a control
      // moved again while this fit was running. That now includes the thread count.
      final String threads = projection.getThreads() > 1
        // Worth saying every time rather than once at startup: with more than one thread the
        // same settings give a different picture on the next run, so two maps being unalike is
        // no longer evidence that anything was changed.
        ? String.format("  [%d threads, embedding not reproducible]", projection.getThreads())
        : "";
      mStatus.setText(String.format("%d points, %d classes, neighbours %d, min dist %.2f, spread %.2f%s",
        projection.getEmbedding().length, projection.getData().getClassNames().length,
        projection.getNeighbours(), projection.getMinDist(), projection.getSpread(), threads));
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
    mElapsed.stop();
    mFitting = false;
    setControlsDisabled(false);
  }

  private void setControlsDisabled(final boolean disabled) {
    mNeighbourSlider.setDisable(disabled);
    mMinDistSlider.setDisable(disabled);
    mSpreadSlider.setDisable(disabled);
    mThreadChoice.setDisable(disabled);
    mRecompute.setDisable(disabled);
    // A source that fixes the point count leaves the slider disabled throughout.
    if (mPointsAdjustable) {
      mPointsSlider.setDisable(disabled);
    }
  }

  /** One immutable set of projection parameters, safe to hand to a background thread. */
  private static final class Settings {
    private final int mNeighbours;
    private final float mMinDist;
    private final float mSpread;
    private final int mThreads;

    private Settings(final int neighbours, final float minDist, final float spread, final int threads) {
      mNeighbours = neighbours;
      mMinDist = minDist;
      mSpread = spread;
      mThreads = threads;
    }

    private Projection project(final PointData data) {
      // A fixed seed on a single thread keeps the picture reproducible; more than one thread
      // makes the embedding differ between runs. One thread is fast enough for everything this
      // viewer offers - all 60 000 MNIST images project in under a minute at the default
      // neighbour count - so the choice is a genuine trade rather than a necessity, which is
      // why it is made in the dropdown and reported in the status line.
      final Umap umap = new Umap()
        .setNumberComponents(2)
        .setNumberNearestNeighbours(mNeighbours)
        .setMinDist(mMinDist)
        .setSpread(mSpread)
        .setSeed(SEED)
        .setThreads(mThreads);
      return new Projection(data, umap.fitTransform(data.getData()), mNeighbours, mMinDist, mSpread, mThreads);
    }
  }
}
