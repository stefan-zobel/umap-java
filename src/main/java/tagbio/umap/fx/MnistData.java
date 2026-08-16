package tagbio.umap.fx;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

/**
 * The MNIST training set in its original IDX form, together with the labels, read straight
 * from the two compressed files as they are distributed.
 *
 * <p>The files are not converted to a delimited format first. All 60 000 images are 9.9 MB
 * like this against roughly 750 MB for the same rows written out as decimal numbers, and
 * reading them avoids parsing 47 million of those.
 *
 * <p><b>Memory.</b> {@link #load} of all 60 000 rows returns a <code>float[60000][784]</code>,
 * which is 188 MB. Every fit reads the data again rather than caching it, so that figure is
 * allocated once per run and the previous one becomes garbage.
 */
final class MnistData implements PointData {

  /** Edge length of an image in pixels. */
  private static final int EDGE = 28;

  /** Pixels per image, 28 by 28. */
  static final int PIXELS = EDGE * EDGE;

  private static final String IMAGE_FILE = "train-images-idx3-ubyte.gz";
  private static final String LABEL_FILE = "train-labels-idx1-ubyte.gz";
  private static final int IMAGE_MAGIC = 2051;
  private static final int LABEL_MAGIC = 2049;
  /** MNIST is the ten decimal digits, whatever subset a given point count happens to contain. */
  private static final int DIGITS = 10;

  private final float[][] mData;
  private final String[] mSampleNames;
  private final String[] mClassNames;
  private final int[] mClassIndices;

  private MnistData(final float[][] data, final String[] sampleNames, final String[] classNames,
    final int[] classIndices) {
    mData = data;
    mSampleNames = sampleNames;
    mClassNames = classNames;
    mClassIndices = classIndices;
  }

  /**
   * Whether a directory holds the two files this reader needs. Used to decide from the launch
   * argument alone whether the viewer is looking at MNIST or at a delimited file.
   *
   * @param directory the candidate directory
   * @return true if both IDX files are present
   */
  static boolean isMnistDirectory(final File directory) {
    return directory.isDirectory()
      && new File(directory, IMAGE_FILE).isFile()
      && new File(directory, LABEL_FILE).isFile();
  }

  /**
   * How many images the file in the given directory holds, read from its header alone. Used to
   * bound the point count the viewer offers rather than trusting the usual 60 000.
   *
   * @param directory the directory holding the two IDX files
   * @return the number of images available
   * @throws IOException if the file is missing or is not an MNIST image file
   */
  static int available(final File directory) throws IOException {
    try (DataInputStream in = open(new File(directory, IMAGE_FILE))) {
      readMagic(in, IMAGE_MAGIC, IMAGE_FILE);
      return in.readInt();
    }
  }

  /**
   * Read the first <code>n</code> images and their labels.
   *
   * @param directory the directory holding the two IDX files
   * @param n how many images to read
   * @return the images, their labels and the derived class names
   * @throws IOException if either file is missing, malformed, or holds fewer than n records
   */
  static MnistData load(final File directory, final int n) throws IOException {
    if (n < 1) {
      throw new IOException("Asked for " + n + " images");
    }
    final byte[] pixels = readPayload(new File(directory, IMAGE_FILE), IMAGE_MAGIC, n, PIXELS);
    final byte[] rawLabels = readPayload(new File(directory, LABEL_FILE), LABEL_MAGIC, n, 1);

    final float[][] images = new float[n][PIXELS];
    final String[] sampleNames = new String[n];
    final int[] classIndices = new int[n];
    for (int image = 0; image < n; ++image) {
      final int base = image * PIXELS;
      final float[] out = images[image];
      for (int i = 0; i < PIXELS; ++i) {
        // Transposed within the image, and 1.0 marks the background rather than the ink. This
        // is the encoding the reference t-SNE demo data was produced with, and it is fixed
        // rather than offered as a control so that a UMAP map and a t-SNE map of MNIST are
        // maps of the same input and can be put side by side.
        final int transposed = (i % EDGE) * EDGE + (i / EDGE);
        out[i] = (pixels[base + transposed] & 0xff) != 0 ? 0.0f : 1.0f;
      }
      final int label = rawLabels[image] & 0xff;
      if (label >= DIGITS) {
        throw new IOException("Label " + label + " at index " + image + " is not a decimal digit");
      }
      // The class index is the digit itself rather than an order of first appearance, so a
      // digit keeps its colour no matter how many points are being shown.
      classIndices[image] = label;
      sampleNames[image] = label + ":" + image;
    }

    final String[] classNames = new String[DIGITS];
    for (int digit = 0; digit < DIGITS; ++digit) {
      classNames[digit] = String.valueOf(digit);
    }
    return new MnistData(images, sampleNames, classNames, classIndices);
  }

  /** Reads the first <code>count</code> records of <code>recordSize</code> bytes each, past the header. */
  private static byte[] readPayload(final File file, final int magic, final int count,
    final int recordSize) throws IOException {
    try (DataInputStream in = open(file)) {
      readMagic(in, magic, file.getName());
      final int records = in.readInt();
      if (records < count) {
        throw new IOException(file + " holds only " + records + " records, asked for " + count);
      }
      // the image file states its two dimensions here, the label file states nothing more
      if (magic == IMAGE_MAGIC) {
        final int rows = in.readInt();
        final int columns = in.readInt();
        if (rows * columns != recordSize) {
          throw new IOException(file + " holds " + rows + " x " + columns + " images");
        }
      }
      final byte[] payload = new byte[count * recordSize];
      in.readFully(payload);
      return payload;
    }
  }

  private static void readMagic(final DataInputStream in, final int expected, final String name)
    throws IOException {
    final int magic = in.readInt();
    if (magic != expected) {
      throw new IOException(name + ": expected magic " + expected + ", found " + magic);
    }
  }

  private static DataInputStream open(final File file) throws IOException {
    if (!file.isFile()) {
      throw new IOException("No such file: " + file.getAbsolutePath());
    }
    return new DataInputStream(new GZIPInputStream(
      new BufferedInputStream(new FileInputStream(file), 1 << 16), 1 << 16));
  }

  @Override
  public float[][] getData() {
    return mData;
  }

  @Override
  public String[] getSampleNames() {
    return mSampleNames;
  }

  @Override
  public String[] getClassNames() {
    return mClassNames;
  }

  @Override
  public int[] getClassIndices() {
    return mClassIndices;
  }
}
