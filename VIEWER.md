# JavaFX viewer

A 2D scatter view of a UMAP projection: each row of a delimited data file becomes one
point, coloured by the class encoded in its sample name. The projection can be recomputed
from the UI with different parameters.

Everything lives in `src/main/java/tagbio/umap/fx/`. Nothing in `tagbio.umap` was changed
for it — the viewer only consumes the public `Umap`, `UmapProgress` and `ProgressListener`
API, so the ported algorithm stays independent of any UI code.

## Running it

```bash
mvn exec:exec
```

```bash
mvn exec:exec "-Dumap.data=src/test/resources/tagbio/umap/iris.tsv"
```

`umap.data` is a POM property defaulting to `src/test/resources/tagbio/umap/digits.tsv`
(1797 points, 10 classes). The exec plugin is configured for `exec:exec`, not `exec:java`:
it invokes `${java.home}/bin/java` directly so that JavaFX can be put on the **module path**
via `--module-path` / `--add-modules javafx.controls,javafx.graphics`.

From the IDE, run **`tagbio.umap.fx.Launcher`** as a Java application with the data file as
the first program argument.

### Why there is a `Launcher`

`Launcher` deliberately does **not** extend `Application`. When JavaFX is a plain classpath
dependency — which is how the IDE resolves the Maven dependencies — a `main` method declared
inside an `Application` subclass aborts with *"JavaFX runtime components are missing"*.
Routing through a class that does not extend `Application` works both there and when JavaFX
is on the module path, so there is a single entry point for both launch paths.

## Input format

The first column is the sample name, the remaining columns are numeric attributes:

```
sample      att0  att1  att2  att3
setosa:0    5.1   3.5   1.4   0.2
setosa:1    4.9   3.0   1.4   0.2
virginica:2 6.3   3.3   6.0   2.5
```

- The part of the sample name **before the first colon** is the class. A name without a
  colon is its own class.
- Tab separated, or comma separated when the file name contains `.csv`.
- `.gz` files are decompressed while reading.
- Class indices are assigned in order of first appearance, so colours are stable for a
  given file.
- At least two attribute columns are required.

Malformed input is skipped rather than fatal: a row whose field count disagrees with the
header, or one containing a value that is not a number, is dropped with a warning on
`System.err`. Only a missing header, fewer than two attribute columns, or a file with no
usable rows at all raise `IOException`.

## Structure

```
Launcher ──▶ UmapViewer ──┬──▶ EmbeddingCanvas ──▶ ViewTransform
                          │                    └──▶ Palette
                          ├──▶ Legend ─────────────▶ Palette
                          ├──▶ LabelledData
                          ├──▶ ParameterRange
                          ├──▶ ImageExport
                          └──▶ Projection
```

| Class | Responsibility |
|---|---|
| `Launcher` | `main`; starts the application without extending `Application` |
| `UmapViewer` | Layout, controls, background fitting, progress, status |
| `EmbeddingCanvas` | Drawing, mouse interaction, hover readout |
| `ViewTransform` | Data coordinates → screen pixels, zoom and pan. No JavaFX types |
| `LabelledData` | Reads a delimited file into values, sample names and class indices |
| `Projection` | An embedding plus the data and the parameters that produced it |
| `ParameterRange` | Which parameter values are safe to offer, and why |
| `Palette` | Colour per class index |
| `Legend` | Class name next to its colour; owns which classes are shown |
| `ImageExport` | Rendered image → PNG file |

## Design decisions

**Canvas, not `ScatterChart`.** A scatter chart allocates a scene graph node per data point.
That is already sluggish at the 1797 points of `digits.tsv` and unusable at the row counts
UMAP is normally applied to. `EmbeddingCanvas` draws into a `Canvas` in a single pass, which
also gives direct control over colour, alpha and point size.

**One scale for both axes.** UMAP embeddings are geometrically meaningful — the distances
between clusters carry information. Fitting x and y independently to the viewport would
stretch one axis and destroy exactly that. `ViewTransform.fit` therefore uses a single
scale, `min(usableWidth / rangeX, usableHeight / rangeY)`, and centres the shorter axis in
the leftover space.

**The transform, explicitly.** `screenX = offsetX + x · scale` and
`screenY = offsetY − y · scale`. The second is subtracted because screen y grows downwards
while the data's does not. Zooming multiplies `scale` and moves the offsets so that the
point under the cursor stays put: `offset' = cursor − (cursor − offset) · factor`, which
happens to take the same form on both axes.

**`ViewTransform` is JavaFX-free on purpose.** It holds the only real arithmetic in the
viewer. Keeping it free of toolkit types means the fit, pan and zoom behaviour can be
asserted in plain unit tests instead of being verified by looking at the window.

**Fitting runs off the application thread.** `Umap.fitTransform` takes seconds on small data
and much longer on real data; on the FX application thread the window would simply freeze
for the duration. It runs in a `javafx.concurrent.Task` on a daemon thread, with a
`ProgressListener` registered on `UmapProgress` feeding a progress bar through
`Platform.runLater`. `UmapProgress` is a global singleton, so the listener is removed again
in both `setOnSucceeded` and `setOnFailed`.

**Slider values are read on the FX thread.** `UmapViewer.Settings` is an immutable snapshot
of the three parameters, taken on the application thread and handed to the background
thread. Reading JavaFX properties from the worker would be a threading violation.

**`Projection` carries its own parameters.** The status line reports what the visible
embedding was computed with, not the current slider positions. Those can differ: if a slider
moves again while a fit is running, the second request is dropped by the `mFitting` guard,
and reporting live slider values would then describe a projection that was never drawn.

**Recompute on release, not on every value.** Dragging a slider emits a value change per
pixel. Recomputation is triggered from `valueChangingProperty` flipping to false, with a
fallback on `valueProperty` for track clicks and arrow keys, which change the value without
a drag. The controls are disabled while a fit runs, so two fits can never overlap on the
shared `UmapProgress`.

**Point size is constant in screen pixels.** Zooming in resolves dense clusters into
individual points instead of magnifying blobs. Points outside the viewport are skipped, so
zooming into a large data set also costs less to draw.

**Picking is a linear scan in screen space.** Screen space keeps the pick radius a constant
10 pixels at every zoom level, and a linear scan costs one subtraction and one multiply per
point, only on mouse movement. No spatial index is warranted at these sizes.

**Class-outer draw loop.** Points are drawn one class at a time so the fill colour is set
once per class instead of once per point. Filtering a class out is then a single check per
class rather than per point.

**Hidden points are not pickable.** The class filter is applied in `pickNearest` as well as
in the draw loop; otherwise the hover readout would describe a point that is not on screen.

**The last visible class cannot be hidden.** Clicking it is ignored, because an entirely
empty plot is indistinguishable from a bug.

**PNG export avoids `SwingFXUtils`.** The usual route from a JavaFX image to a file goes
through `SwingFXUtils`, which lives in the separate `javafx-swing` artifact. Rather than add
a fourth platform-classified dependency and another `--add-modules` entry, `ImageExport`
copies the pixels out through a `PixelReader` into an `int[]` and hands that to a
`BufferedImage`, which the JDK's own `ImageIO` writes. As a side effect the JDK half of the
conversion has no JavaFX types in it and can be tested directly.

## Parameters

| Control | Range | Default | Effect |
|---|---|---|---|
| Neighbours | 2 – 100 | 15 | Local vs. global structure |
| Min dist | 0 – `maxMinDist(spread)` | 0.10 | How tightly points may pack |
| Spread | 0.50 – 1.50 | 1.00 | Overall scale of the layout |

Sliders step in 0.05 (1 for neighbours). Neighbour counts need no clamping to the data size:
`Umap` itself reduces the value to `rows - 1` for small inputs (`Umap.java:1092`).

### Why min dist is capped by spread

This is not a cosmetic limit. The Java port replaces scipy's `curve_fit` with interpolation
in a fixed lookup table in `Curve.java`, and **the argument check there is not sufficient**.
It permits `0 <= minDist <= spread`, but the table row for a spread is selected by
`(int) (10 * spread)` and holds `2 * (index + 1)` entries, while the interpolation also reads
the entry *after* the one it lands on. The distance index is `(int) (20 * minDist)`, so the
real constraint is:

```
(int) (20 * minDist) <= 2 * (int) (10 * spread)
```

A pair like `spread = 0.55, minDist = 0.55` passes the argument check and then fails with
`ArrayIndexOutOfBoundsException`. `ParameterRange.maxMinDist` rounds the spread down to a
tenth, which satisfies the constraint for every spread and is exact when the spread is
itself a multiple of a tenth. The index is computed in `float` so that it matches what
`curveFit` sees. `ParameterRangeTest` sweeps every reachable slider combination against the
real `Curve.curveFit` and also pins the failing case, so the bound cannot silently rot.

Moving the spread slider updates the minimum distance slider's maximum live. An `mAdjusting`
flag suppresses the recomputation that the resulting clamp would otherwise trigger.

### Reproducibility

The viewer fixes `setSeed(42)` and `setThreads(1)`. Single threaded with a fixed seed the
embedding is deterministic, so the same file and parameters always give the same picture.
`setThreads(n > 1)` would be faster but produces a different embedding on every run, which
makes visual comparison meaningless. If speed matters more than reproducibility for large
data, that is the line to change — in `UmapViewer.Settings.project`.

## Interaction

| Input | Action |
|---|---|
| Scroll wheel | Zoom about the cursor, 1.15× per notch |
| Drag | Pan |
| Hover | Readout with sample name, class and coordinates |
| Click a legend row | Hide or show that class |
| Show all | Bring every hidden class back; disabled when nothing is hidden |
| Reset view | Refit the whole embedding |
| Export PNG… | Save the plot as it currently looks |

Zoom is bounded to 0.25× – 400× the fitted scale. Resizing the window refits the embedding,
unless zoom or pan has been used — in that case the current view is kept so the user does
not lose their place.

A hidden class is dimmed in the legend and its swatch is drawn as an outline instead of a
filled square. The filter is a view setting only: it never triggers a recomputation, and it
is cleared whenever a new projection arrives, since the set of classes can change with the
data.

The PNG contains the plot area only — not the legend, the sliders or the status bar — at
whatever zoom, pan and class filter are active. Its resolution is the on-screen size of the
plot, so enlarge the window for a larger image. The hover readout is a sibling of the canvas
rather than part of it, so it is never captured even if it happens to be visible.

## Colours

`Palette` holds 12 colours chosen to stay distinguishable against a light background. Beyond
that, hues step by the golden angle (137.508°) so consecutive class indices land far apart
in colour rather than forming a gradient. Points are drawn at 0.75 alpha, because UMAP
clusters overlap heavily and opaque discs hide the density.

## Tests

| Test | Covers |
|---|---|
| `tagbio.umap.fx.ViewTransformTest` | Fit bounds and centring, single shared scale, y flip, pan precision, zoom pinning the point under the cursor, zoom limits, degenerate ranges |
| `tagbio.umap.fx.LabelledDataTest` | TSV/CSV/gzip reading, class order and indices, skipped rows, the four failure modes |
| `tagbio.umap.fx.ImageExportTest` | ARGB → `BufferedImage` conversion, rejection of short pixel arrays, PNG signature, lossless pixel round trip through `ImageIO` |
| `tagbio.umap.ParameterRangeTest` | Every reachable slider combination against `Curve.curveFit`, plus the case the bound exists to prevent |

`ParameterRangeTest` sits in `tagbio.umap` rather than the `fx` test package because `Curve`
is package private. All four are registered in `AllTests.suite()`.

37 tests across the four classes; 118 for the full suite. Note that `mvn test` does not
currently work from the command line — the Maven super-POM's default surefire version does
not run under JDK 25 — so the suite is run from the IDE.

## Build

JavaFX 25.0.3 is declared as three separate dependencies (`javafx-base`, `javafx-graphics`,
`javafx-controls`), each with an explicit `${javafx.platform}` classifier. The classifier
cannot be left to transitive resolution: the transitive dependencies of `javafx-controls`
resolve *without* a classifier and would pull in the empty platform-independent jars. Any
further JavaFX module has to be declared the same way.

`javafx.platform` is set by OS-activated profiles (`win`, `mac`, `mac-aarch64`, `linux`),
with `win` in `<properties>` as the fallback when no profile matches.

Files in this package carry **no license header**, unlike the ported core.

## Not implemented

- Continuous colour scales — labels are treated as categorical only.
- Export at a resolution independent of the window. A `Canvas` is a raster surface, so
  scaling the snapshot would only upscale pixels; a true high resolution export means
  resizing the canvas and redrawing.
- Including the legend in the exported image.
- Any use of `Umap.transform` to place new points into an existing embedding; that method
  is documented as alpha upstream.
- Selecting a different data file from within the UI; the path is a launch argument.
