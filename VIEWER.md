# JavaFX viewer

A 2D scatter view of a UMAP projection: each row of the input becomes one point, coloured by
its class. The projection can be recomputed from the UI with different parameters.

The points come from one of two places, decided from the launch argument alone: a **delimited
data file**, where the class is the sample name prefix, or a **directory holding the MNIST IDX
pair**, where the class is the digit.

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

```bash
mvn exec:exec "-Dumap.data=src/test/resources/datasets" "-Dumap.heap=-Xmx8g"
```

```bash
mvn exec:exec "-Dumap.data=src/test/resources/datasets" "-Dumap.threads=12"
```

`umap.data` is a POM property defaulting to `src/test/resources/tagbio/umap/digits.tsv`
(1797 points, 10 classes). Point it at a **directory** containing
`train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` to read MNIST instead; anything
else is treated as a delimited file. The exec plugin is configured for `exec:exec`, not `exec:java`:
it invokes `${java.home}/bin/java` directly so that JavaFX can be put on the **module path**
via `--module-path` / `--add-modules javafx.controls,javafx.graphics`.

From the IDE, run **`tagbio.umap.fx.Launcher`** as a Java application with the data file or
the MNIST directory as the first program argument.

### Why there is a `Launcher`

`Launcher` deliberately does **not** extend `Application`. When JavaFX is a plain classpath
dependency — which is how the IDE resolves the Maven dependencies — a `main` method declared
inside an `Application` subclass aborts with *"JavaFX runtime components are missing"*.
Routing through a class that does not extend `Application` works both there and when JavaFX
is on the module path, so there is a single entry point for both launch paths.

## Input

Both sources arrive behind `PointData` — values, sample names, class names, class index per
row — so the drawing, the legend and the projection never learn which one they got.

### Delimited files

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

### MNIST

The two files in their original IDX form, still compressed, read directly — no conversion
step. All 60 000 images are 9.9 MB like this against roughly 750 MB for the same rows written
out as decimal numbers, and reading them avoids parsing 47 million of those.

The class index of a point is **the digit itself**, not an order of first appearance, so a
digit keeps its colour however many points are shown; the legend therefore always lists ten
classes. The pixel encoding is fixed: transposed within the image and inverted so the
background is 1.0, which is what the reference t-SNE demo data was produced with. That makes a
UMAP map and a t-SNE map of MNIST maps of the same input, which is the only reason to have a
choice here at all — so there is no control for it.

## Structure

```
Launcher ──▶ UmapViewer ──┬──▶ EmbeddingCanvas ──▶ ViewTransform
                          │                    └──▶ Palette
                          ├──▶ Legend ─────────────▶ Palette
                          ├──▶ PointData ──┬──▶ LabelledData
                          │                └──▶ MnistData
                          ├──▶ ParameterRange
                          ├──▶ ImageExport
                          └──▶ Projection
```

| Class | Responsibility |
|---|---|
| `Launcher` | `main`; starts the application without extending `Application` |
| `UmapViewer` | Layout, controls, source selection, background fitting, progress, status |
| `EmbeddingCanvas` | Drawing, mouse interaction, hover readout |
| `ViewTransform` | Data coordinates → screen pixels, zoom and pan. No JavaFX types |
| `PointData` | Values, sample names and classes, whatever they were read from |
| `LabelledData` | Reads a delimited file into `PointData` |
| `MnistData` | Reads the MNIST IDX pair into `PointData` |
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

**The progress bar is indeterminate, and the status line counts seconds.** This is not
laziness, it is the only honest reading of what `UmapProgress` reports. The total is built in
three goes — `reset(5)`, then the trees and descent iterations, then the epochs
(`Umap.java:1103`, `:232`, `:1191`) — so the *denominator grows twice during a run* while the
count is monotonic. A determinate bar therefore visibly runs backwards, which it was observed
doing. Worse, the fraction also stops: measured on 60 000 points at 50 neighbours, a 157 s run
leaves it on one value for 42 s and on the next for 23 s, because the graph construction
between the neighbour search and the epochs reports twice in a minute of work. Two thirds of a
minute of a motionless bar is indistinguishable from a hang, and both symptoms were reported as
exactly that.

A barber pole can say "still working" truthfully for the whole run, which is the one thing that
is always true, so that is what the bar does. The numbers go in the status line — `step 21/40,
47 s` — where a growing denominator is legible instead of mysterious, and a `Timeline` ticks the
seconds once a second so the display keeps moving through phases that report nothing at all.
None of this needed a change in `tagbio.umap`.

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
point, only on mouse movement. At 60 000 points this is the one place where a spatial index
would eventually pay; it is not there yet.

**Class-outer draw loop, over grouped indices.** Points are drawn one class at a time so the
fill colour is set once per class instead of once per point. Filtering a class out is then a
single check per class rather than per point. The point indices are bucketed by class once in
`setProjection`, which turns the loop from a full scan per class into a single pass: at the ten
classes and 60 000 points of MNIST, 60 000 iterations per redraw instead of 600 000. That did
not matter at the 1797 points of `digits.tsv` and does at sixty thousand.

**The data is read again for every fit.** Re-reading is well under a second either way, against
a projection measured in seconds to a minute. Caching it would mean deciding when the cached
table has gone stale against the point count, which is more machinery than the saving is worth.

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
| Points | 5 000 – 60 000 | 10 000 | How many MNIST images to project |

Sliders step in 0.05 (1 for neighbours, 5000 for points). Neighbour counts need no clamping to
the data size: `Umap` itself reduces the value to `rows - 1` for small inputs
(`Umap.java:1092`).

The point slider only appears to do anything for MNIST. A delimited file is projected whole, so
the slider is disabled and reads `file`.

### Why the point count starts at 5000

Not an arbitrary floor, and not a performance guess. `Umap` computes the **full pairwise
distance matrix** below `SMALL_PROBLEM_THRESHOLD`, which is 4096 rows (`Umap.java:33`), and the
approximate nearest neighbour descent above it. Measured on MNIST pixels, single threaded, at
the settings this viewer fixes:

| Points | Path | Time |
|---:|---|---:|
| 1 000 | pairwise | 1.3 s |
| 2 500 | pairwise | 3.1 s |
| 4 000 | pairwise | 6.9 s |
| 5 000 | nn-descent | **5.0 s** |
| 10 000 | nn-descent | 10.0 s |
| 20 000 | nn-descent | 12.7 s |
| 60 000 | nn-descent | 54.3 s |

4000 points cost more than 5000 do — more points, less time — and the distance matrix alone is
67 MB just under the threshold. Starting at 5000 keeps the whole offered range on the side
where the viewer gets steadily slower the more it is given, instead of hiding its worst case in
the middle of the slider.

The flat stretch between 10 000 and 20 000 is the epoch count dropping from 500 to 200 above
10 000 graph rows (`Umap.java:599-602`); past that it is linear again.

These are single figures on a loaded desktop, not averages, and they scatter accordingly: the
60 000 row was 54.3 s here and 44.9 s in a later batch on a quieter machine. Compare runs from
one batch, never across sessions. The shape is what matters, and the shape is stable.

All of the above is at the default 15 neighbours. **The neighbour count multiplies these
times**, because it sets how many edges the graph has and the layout walks every edge on every
epoch. The same 60 000 points at 50 neighbours take **157 s** against 54 s — about the ratio of
the neighbour counts. That is the slowest thing the viewer can be asked for, and it is still
under three minutes, so no bound is imposed on the combination; the status line just has to be
honest about the wait, which is what the elapsed seconds are for.

Memory is not the constraint at any of this. The 60 000 by 50 run peaks at 1451 MB, well inside
the 4 GB `umap.heap` default.

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

The viewer fixes `setSeed(42)` and defaults to one thread. Single threaded with a fixed seed the
embedding is deterministic, so the same input and parameters always give the same picture.

The measurements above are what makes that an easy default rather than a compromise: all 60 000
MNIST images project in under a minute on one thread, so nothing the viewer offers is slow
enough that an unstable picture is obviously worth it.

### `umap.threads`

More threads can be asked for through the POM property, which is passed to the application as a
system property of the same name:

| `umap.threads` | Threads used |
|---|---|
| unset, or `<= 0` | 1 |
| `> 0` | `min(umap.threads, logical cores)` |

The request is honoured up to what the machine has, so asking for more threads than there are
cores gets the core count rather than oversubscribing it. A value that does not parse is treated
as unset rather than as a reason to refuse to start. `UmapViewer.resolveThreads` takes the core
count as an argument so `ThreadResolutionTest` can assert the rule without depending on the
machine it runs on.

It costs reproducibility outright: two consecutive runs of one configuration on 12 threads were
compared coordinate by coordinate and are not equal. That is why the status line appends
`[N threads, embedding not reproducible]` after every fit rather than mentioning it once at
startup — without it, two maps looking unalike would wrongly suggest that something had been
changed between them.

And it buys about **1.6x**, no matter how many cores are thrown at it. That is worth
understanding before turning it on.

#### Why 1.6x and not 12x

`setThreads` reaches exactly one thing. The whole core contains two parallel constructs —
`ParallelNearestNeighborDescent` and the forest in `RandomProjectionTree` — and both live inside
`nearestNeighbors`. `fuzzySimplicialSet` takes a `threads` argument but only forwards it, and
only when the neighbours have not already been computed; on the path the viewer takes they have
(`Umap.java:1158`), so it is dead there. Everything after it — `smoothKnnDist`,
`computeMembershipStrengths`, the transpose products, `eliminateZeros` — and the entire epoch
layout are single threaded, and the layout is never even offered the parameter.

Measured on 60 000 points at 15 neighbours, 12 logical cores, both runs in one batch:

| Phase | 1 thread | 12 threads | |
|---|---:|---:|---:|
| setup | 0.8 s | 1.2 s | 0.7x |
| `nearestNeighbors` | 23.7 s | 5.9 s | **4.0x** |
| graph construction + epochs | 20.3 s | 20.0 s | **1.0x** |
| total | 44.9 s | 27.2 s | **1.65x** |

The serial phase is unchanged to within the noise, which is the measurement confirming what the
code says. The parallel phase does scale — 4.0x here, 4.8x at 20 000 points — so the threading
itself is not the problem. It is simply that only about half the run is parallel at all.

That halves-and-halves split puts a hard ceiling on the whole thing: with infinitely many cores
the serial 20.0 s and the setup remain, so the best achievable is `44.9 / 21.2` ≈ **2.1x**. The
observed 1.65x is about 79% of that. There is nothing left on the table worth chasing.

**More neighbours make threading less useful, not more.** The neighbour count grows the graph,
which grows exactly the serial half — the matrix work that builds it and the edges every epoch
walks. At 60 000 points and 50 neighbours the serial phases are roughly three quarters of the
run rather than half, so the ceiling there is nearer 1.3x. The setting most likely to make
somebody reach for more threads is the one that benefits least from them.

Lifting any of this means parallelising the layout inside `tagbio.umap`, which is not something
the viewer has ever done to the algorithm.

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
| `tagbio.umap.fx.MnistDataTest` | IDX parsing, source detection, the pixel encoding, class indices independent of the point count, six failure modes |
| `tagbio.umap.fx.ImageExportTest` | ARGB → `BufferedImage` conversion, rejection of short pixel arrays, PNG signature, lossless pixel round trip through `ImageIO` |
| `tagbio.umap.fx.ThreadResolutionTest` | The `umap.threads` rule, including the floor behaviour, an absent property and a malformed one |
| `tagbio.umap.ParameterRangeTest` | Every reachable slider combination against `Curve.curveFit`, plus the case the bound exists to prevent |

`ParameterRangeTest` sits in `tagbio.umap` rather than the `fx` test package because `Curve`
is package private. All six are registered in `AllTests.suite()`.

`MnistDataTest` writes its own IDX files into a temporary directory rather than reading the
real ones, so the suite runs whether or not MNIST has been copied into this project.

63 tests across the six classes; 144 for the full suite. Note that `mvn test` does not
currently work from the command line — the Maven super-POM's default surefire version does
not run under JDK 25 — so the suite is run from the IDE.

Three tests in `UmapTest` (`testIris`, `testIrisViaDouble`, `testSmoothKnnDist`) fail, and have
done since before the viewer existed. They are assertions about the ported algorithm itself,
not about anything here; the same three fail with the same values on a checkout with no viewer
in it at all.

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

- Cancelling a run. There is no abort hook anywhere in `tagbio.umap`, and adding one would mean
  changing the algorithm, which the viewer has so far never done. At the measured times a
  progress bar carries the wait instead.
- Continuous colour scales — labels are treated as categorical only.
- Export at a resolution independent of the window. A `Canvas` is a raster surface, so
  scaling the snapshot would only upscale pixels; a true high resolution export means
  resizing the canvas and redrawing.
- Including the legend in the exported image.
- Any use of `Umap.transform` to place new points into an existing embedding; that method
  is documented as alpha upstream.
- Selecting a different data file from within the UI; the path is a launch argument.
- Any MNIST-like source other than the training set pair, and any pixel encoding other than the
  one fixed in `MnistData`.
